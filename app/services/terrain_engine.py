import importlib
import io
import logging
import math
import os
from dataclasses import dataclass
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from joblib import Parallel, delayed
from PIL import Image
from rasterio.transform import from_bounds
from rasterio.warp import reproject, Resampling

from app.models.CoveragePredictionRequest import CoveragePredictionRequest


logger = logging.getLogger(__name__)


@dataclass
class DemSource:
    name: str
    path: str
    dataset: rasterio.io.DatasetReader


class TerrainEngine:
    """
    Terrain-based coverage engine that uses local DEM GeoTIFFs and
    a line-of-sight + free-space path loss model.

    This engine supports multi-core processing and optional CUDA acceleration
    (if CuPy is installed) for the distance/path loss computations.
    """

    def __init__(
        self,
        dem_1m_path: Optional[str],
        dem_10m_path: Optional[str],
        max_pixels: int = 2048,
        azimuth_step: int = 1,
        blocked_loss_db: float = 20.0,
    ) -> None:
        if not dem_1m_path and not dem_10m_path:
            raise ValueError("At least one DEM path must be provided via DEM_1M_PATH or DEM_10M_PATH.")

        self.dem_1m = self._load_dem("1m", dem_1m_path) if dem_1m_path else None
        self.dem_10m = self._load_dem("10m", dem_10m_path) if dem_10m_path else None

        self.max_pixels = max_pixels
        self.azimuth_step = azimuth_step
        self.blocked_loss_db = blocked_loss_db
        self.worker_count = max(1, int(os.getenv("PROPAGATION_WORKERS", os.cpu_count() or 1)))
        self.use_cuda = self._detect_cuda()

        logger.info(
            "Initialized TerrainEngine with 1m DEM=%s, 10m DEM=%s, max_pixels=%s, workers=%s, cuda=%s",
            bool(self.dem_1m),
            bool(self.dem_10m),
            self.max_pixels,
            self.worker_count,
            self.use_cuda,
        )

    def coverage_prediction(self, request: CoveragePredictionRequest) -> bytes:
        dem = self._select_dem(request)
        dem_data, transform = self._read_dem_window(dem, request)

        if dem_data.size == 0:
            raise RuntimeError("DEM window is empty for the requested area.")

        tx_elevation = self._sample_tx_elevation(dem_data, transform, request.lat, request.lon)
        signal_dbm = self._compute_signal_strength(
            dem_data,
            transform,
            request,
            tx_elevation,
        )

        return self._create_geotiff(signal_dbm, transform, request)

    def coverage_prediction_with_kmz(self, request: CoveragePredictionRequest) -> Tuple[bytes, bytes]:
        dem = self._select_dem(request)
        dem_data, transform = self._read_dem_window(dem, request)

        if dem_data.size == 0:
            raise RuntimeError("DEM window is empty for the requested area.")

        tx_elevation = self._sample_tx_elevation(dem_data, transform, request.lat, request.lon)
        signal_dbm = self._compute_signal_strength(
            dem_data,
            transform,
            request,
            tx_elevation,
        )

        geotiff_data = self._create_geotiff(signal_dbm, transform, request)
        kmz_data = self._create_kmz(signal_dbm, transform, request)
        return geotiff_data, kmz_data

    def _load_dem(self, name: str, path: str) -> DemSource:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{name} DEM path does not exist: {path}")

        dataset = rasterio.open(path)
        logger.info("Loaded %s DEM from %s (CRS=%s)", name, path, dataset.crs)
        return DemSource(name=name, path=path, dataset=dataset)

    def _select_dem(self, request: CoveragePredictionRequest) -> DemSource:
        preferred = self.dem_1m if request.high_resolution else self.dem_10m
        fallback = self.dem_10m if request.high_resolution else self.dem_1m

        if preferred and self._covers_location(preferred.dataset, request.lat, request.lon):
            return preferred

        if fallback and self._covers_location(fallback.dataset, request.lat, request.lon):
            logger.info("Falling back to %s DEM for %s", fallback.name, request)
            return fallback

        available = [dem.name for dem in [self.dem_1m, self.dem_10m] if dem]
        raise RuntimeError(
            f"No DEM coverage available for {request.lat}, {request.lon}. "
            f"Available datasets: {', '.join(available)}"
        )

    def _covers_location(self, dataset: rasterio.io.DatasetReader, lat: float, lon: float) -> bool:
        if not dataset.crs:
            return False

        if dataset.crs.to_epsg() == 4326:
            return (
                dataset.bounds.left <= lon <= dataset.bounds.right
                and dataset.bounds.bottom <= lat <= dataset.bounds.top
            )

        xs, ys = rasterio.warp.transform("EPSG:4326", dataset.crs, [lon], [lat])
        x = xs[0]
        y = ys[0]
        return dataset.bounds.left <= x <= dataset.bounds.right and dataset.bounds.bottom <= y <= dataset.bounds.top

    def _read_dem_window(
        self, dem: DemSource, request: CoveragePredictionRequest
    ) -> Tuple[np.ndarray, rasterio.Affine]:
        lat, lon, radius_m = request.lat, request.lon, request.radius

        lat_deg_per_m = 1.0 / 111_320.0
        lon_deg_per_m = 1.0 / (111_320.0 * max(math.cos(math.radians(lat)), 0.1))

        delta_lat = radius_m * lat_deg_per_m
        delta_lon = radius_m * lon_deg_per_m
        lat_min = lat - delta_lat
        lat_max = lat + delta_lat
        lon_min = lon - delta_lon
        lon_max = lon + delta_lon

        resolution_m = self._estimate_resolution_m(dem.dataset, lat)
        resolution_m = max(resolution_m, (radius_m * 2) / self.max_pixels)

        width = max(1, int((lon_max - lon_min) / (resolution_m * lon_deg_per_m)))
        height = max(1, int((lat_max - lat_min) / (resolution_m * lat_deg_per_m)))

        transform = from_bounds(lon_min, lat_min, lon_max, lat_max, width, height)
        data = np.empty((height, width), dtype=np.float32)

        reproject(
            source=rasterio.band(dem.dataset, 1),
            destination=data,
            src_transform=dem.dataset.transform,
            src_crs=dem.dataset.crs,
            dst_transform=transform,
            dst_crs="EPSG:4326",
            resampling=Resampling.bilinear,
        )

        return data, transform

    def _estimate_resolution_m(self, dataset: rasterio.io.DatasetReader, lat: float) -> float:
        if dataset.crs and dataset.crs.to_epsg() == 4326:
            lat_deg_per_m = 1.0 / 111_320.0
            lon_deg_per_m = 1.0 / (111_320.0 * max(math.cos(math.radians(lat)), 0.1))
            res_deg = max(abs(dataset.res[0]), abs(dataset.res[1]))
            return max(res_deg / lat_deg_per_m, res_deg / lon_deg_per_m)
        return max(abs(dataset.res[0]), abs(dataset.res[1]))

    def _sample_tx_elevation(
        self, dem_data: np.ndarray, transform: rasterio.Affine, lat: float, lon: float
    ) -> float:
        col, row = (~transform) * (lon, lat)
        col = int(np.clip(round(col), 0, dem_data.shape[1] - 1))
        row = int(np.clip(round(row), 0, dem_data.shape[0] - 1))
        return float(dem_data[row, col])

    def _compute_signal_strength(
        self,
        dem_data: np.ndarray,
        transform: rasterio.Affine,
        request: CoveragePredictionRequest,
        tx_elevation: float,
    ) -> np.ndarray:
        rows, cols = np.indices(dem_data.shape)
        lon = transform.c + cols * transform.a + rows * transform.b
        lat = transform.f + cols * transform.d + rows * transform.e

        if self.use_cuda:
            xp = importlib.import_module("cupy")
        else:
            xp = np

        lat_rad = xp.radians(lat)
        lon_rad = xp.radians(lon)
        tx_lat_rad = math.radians(request.lat)
        tx_lon_rad = math.radians(request.lon)

        delta_lat = lat_rad - tx_lat_rad
        delta_lon = lon_rad - tx_lon_rad

        a = xp.sin(delta_lat / 2) ** 2 + xp.cos(tx_lat_rad) * xp.cos(lat_rad) * xp.sin(delta_lon / 2) ** 2
        distance = 2 * 6371000.0 * xp.arcsin(xp.sqrt(a))
        distance = xp.maximum(distance, 1.0)

        azimuth = (
            xp.degrees(
                xp.arctan2(
                    xp.sin(delta_lon) * xp.cos(lat_rad),
                    xp.cos(tx_lat_rad) * xp.sin(lat_rad)
                    - xp.sin(tx_lat_rad) * xp.cos(lat_rad) * xp.cos(delta_lon),
                )
            )
            + 360.0
        ) % 360.0

        if self.use_cuda:
            distance = distance.get()
            azimuth = azimuth.get()

        flat_distance = distance.ravel()
        flat_azimuth = azimuth.ravel()
        flat_dem = dem_data.ravel()

        bin_ids = (flat_azimuth // self.azimuth_step).astype(int)
        num_bins = int(360 / self.azimuth_step)

        blocked = np.zeros_like(flat_distance, dtype=bool)
        tx_height_m = tx_elevation + request.tx_height
        rx_height_m = request.rx_height

        def process_bin(bin_id: int) -> Tuple[np.ndarray, np.ndarray]:
            indices = np.where(bin_ids == bin_id)[0]
            if indices.size == 0:
                return indices, np.array([], dtype=bool)

            distances = flat_distance[indices]
            elevations = flat_dem[indices]

            order = np.argsort(distances)
            ordered_dist = distances[order]
            ordered_elev = elevations[order]

            angles = np.arctan2((ordered_elev + rx_height_m) - tx_height_m, ordered_dist)
            max_angles = np.maximum.accumulate(angles)
            blocked_sorted = angles < max_angles

            blocked_mask = np.zeros_like(distances, dtype=bool)
            blocked_mask[order] = blocked_sorted
            return indices, blocked_mask

        results = Parallel(n_jobs=self.worker_count, prefer="threads")(
            delayed(process_bin)(bin_id) for bin_id in range(num_bins)
        )

        for indices, blocked_mask in results:
            if indices.size:
                blocked[indices] = blocked_mask

        distance_km = flat_distance / 1000.0
        fspl = 32.44 + 20 * np.log10(distance_km) + 20 * np.log10(request.frequency_mhz)

        signal_dbm = (
            request.tx_power
            + request.tx_gain
            + request.rx_gain
            - request.system_loss
            - fspl
            - (blocked.astype(float) * self.blocked_loss_db)
        )

        return signal_dbm.reshape(dem_data.shape)

    def _create_geotiff(
        self,
        signal_dbm: np.ndarray,
        transform: rasterio.Affine,
        request: CoveragePredictionRequest,
    ) -> bytes:
        min_dbm = request.min_dbm
        max_dbm = request.max_dbm
        nodata_value = 255

        normalized = (signal_dbm - min_dbm) / (max_dbm - min_dbm)
        normalized = np.clip(normalized, 0, 1)

        raster = (normalized * 254).round().astype(np.uint8)
        raster = np.where(signal_dbm < request.signal_threshold, nodata_value, raster)

        cmap = plt.get_cmap(request.colormap, 255)
        colors = (cmap(np.linspace(0, 1, 255))[:, :3] * 255).astype(np.uint8)
        colormap = {idx: tuple(color) + (255,) for idx, color in enumerate(colors)}

        with io.BytesIO() as buffer:
            with rasterio.open(
                buffer,
                "w",
                driver="GTiff",
                height=raster.shape[0],
                width=raster.shape[1],
                count=1,
                dtype="uint8",
                crs="EPSG:4326",
                transform=transform,
                photometric="palette",
                compress="lzw",
                nodata=nodata_value,
            ) as dst:
                dst.write(raster, 1)
                dst.write_colormap(1, colormap)

            buffer.seek(0)
            return buffer.read()

    def _create_kmz(
        self,
        signal_dbm: np.ndarray,
        transform: rasterio.Affine,
        request: CoveragePredictionRequest,
    ) -> bytes:
        import zipfile

        min_dbm = request.min_dbm
        max_dbm = request.max_dbm
        nodata_value = 255

        normalized = (signal_dbm - min_dbm) / (max_dbm - min_dbm)
        normalized = np.clip(normalized, 0, 1)
        raster = (normalized * 254).round().astype(np.uint8)
        raster = np.where(signal_dbm < request.signal_threshold, nodata_value, raster)

        cmap = plt.get_cmap(request.colormap)
        cmap_norm = plt.Normalize(vmin=min_dbm, vmax=max_dbm)
        rgba = (cmap(cmap_norm(signal_dbm)) * 255).astype(np.uint8)
        rgba[raster == nodata_value, 3] = 0

        overlay = Image.fromarray(rgba, mode="RGBA")
        with io.BytesIO() as overlay_buffer:
            overlay.save(overlay_buffer, format="PNG")
            overlay_bytes = overlay_buffer.getvalue()

        west, south = transform * (0, raster.shape[0])
        east, north = transform * (raster.shape[1], 0)

        kml = f"""<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
  <Document>
    <name>Meshtastic Coverage</name>
    <GroundOverlay>
      <name>Coverage</name>
      <Icon>
        <href>files/overlay.png</href>
      </Icon>
      <LatLonBox>
        <north>{north}</north>
        <south>{south}</south>
        <east>{east}</east>
        <west>{west}</west>
      </LatLonBox>
    </GroundOverlay>
  </Document>
</kml>
"""

        with io.BytesIO() as kmz_buffer:
            with zipfile.ZipFile(kmz_buffer, "w", zipfile.ZIP_DEFLATED) as kmz:
                kmz.writestr("doc.kml", kml)
                kmz.writestr("files/overlay.png", overlay_bytes)
            kmz_buffer.seek(0)
            return kmz_buffer.read()

    def _detect_cuda(self) -> bool:
        return importlib.util.find_spec("cupy") is not None
