from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import xarray as xr

SENTINEL_THRESHOLD = -100.0
DEFAULT_CMAP = "turbo"
DEFAULT_VMIN = -2.0
DEFAULT_VMAX = 35.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize HadISST SST as map, animation, or point time series."
    )
    parser.add_argument(
        "--mode",
        choices=["map", "animate", "series"],
        default="map",
        help="map: single month map; animate: time range animation; series: point time series.",
    )
    parser.add_argument(
        "--file",
        default="HadISST_sst.nc",
        help="Path to the netCDF file.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output path for image/animation. Default is mode-specific in outputs/.",
    )
    parser.add_argument(
        "--cmap",
        default=DEFAULT_CMAP,
        help="Matplotlib colormap name.",
    )
    parser.add_argument("--vmin", type=float, default=DEFAULT_VMIN, help="Colorbar min.")
    parser.add_argument("--vmax", type=float, default=DEFAULT_VMAX, help="Colorbar max.")
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display figure window after saving (map/series mode).",
    )

    # map mode下的可配置项
    parser.add_argument(
        "--time",
        default=None,
        help="Target time for map mode, e.g. 2000-01 or 2000-01-16.",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=None,
        help="Zero-based time index for map mode when --time is not provided.",
    )

    # animate mode下的可配置项
    parser.add_argument("--start", default=None, help="Animation start time, e.g. 1990-01.")
    parser.add_argument("--end", default=None, help="Animation end time, e.g. 1995-12.")
    parser.add_argument(
        "--step",
        type=int,
        default=1,
        help="Use every N-th month in animation.",
    )
    parser.add_argument("--fps", type=int, default=6, help="Animation frames per second.")
    parser.add_argument(
        "--max-frames",
        type=int,
        default=240,
        help="Safety cap for animation frame count.",
    )

    # series mode下的可配置项
    parser.add_argument("--lon", type=float, default=None, help="Point longitude for series mode.")
    parser.add_argument("--lat", type=float, default=None, help="Point latitude for series mode.")
    parser.add_argument(
        "--rolling",
        type=int,
        default=12,
        help="Rolling mean window in months for series mode. Set <=1 to disable.",
    )
    parser.add_argument(
        "--csv-output",
        default=None,
        help="Optional CSV path for series mode. Default: same name as figure with .csv suffix.",
    )

    return parser.parse_args()


def normalize_time_string(time_text: str) -> str:
    text = time_text.strip()
    if len(text) == 7:
        return f"{text}-15"
    return text


def to_datetime64(time_text: str) -> np.datetime64:
    return np.datetime64(normalize_time_string(time_text))


def sanitize_text(raw: str) -> str:
    return raw.replace(":", "-").replace(" ", "_")


def get_masked_sst(ds: xr.Dataset) -> xr.DataArray:
    sst = ds["sst"]
    # HadISST有一些没意义的数据，比如-1000
    return sst.where(sst > SENTINEL_THRESHOLD)


def select_single_time(sst: xr.DataArray, time_text: str | None, idx: int | None) -> xr.DataArray:
    if time_text:
        target = to_datetime64(time_text)
        return sst.sel(time=target, method="nearest")

    if idx is None:
        idx = -1
    return sst.isel(time=idx)


def select_time_range(sst: xr.DataArray, start: str | None, end: str | None) -> xr.DataArray:
    start_dt = to_datetime64(start) if start else sst["time"].values[0]
    end_dt = to_datetime64(end) if end else sst["time"].values[-1]
    return sst.sel(time=slice(start_dt, end_dt))


def default_map_output(selected_date: str) -> Path:
    out_dir = Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"sst_map_{sanitize_text(selected_date)}.png"


def default_animation_output(start_date: str, end_date: str, step: int) -> Path:
    out_dir = Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / (
        f"sst_anim_{sanitize_text(start_date)}_to_{sanitize_text(end_date)}_step{step}.gif"
    )


def default_series_output(lon: float, lat: float) -> Path:
    out_dir = Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"sst_series_lon{lon:.1f}_lat{lat:.1f}.png"


def plot_single_map(ds: xr.Dataset, args: argparse.Namespace, plt) -> None:
    sst = get_masked_sst(ds)
    selected = select_single_time(sst, args.time, args.index)

    selected_date = np.datetime_as_string(selected["time"].values, unit="D")
    output_path = Path(args.output) if args.output else default_map_output(selected_date)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lon = ds["longitude"].values
    lat = ds["latitude"].values

    fig, ax = plt.subplots(figsize=(12, 5), dpi=150)
    mesh = ax.pcolormesh(
        lon,
        lat,
        selected.values,
        shading="auto",
        cmap=args.cmap,
        vmin=args.vmin,
        vmax=args.vmax,
    )

    cbar = fig.colorbar(mesh, ax=ax, pad=0.02)
    cbar.set_label("SST (degC)")
    ax.set_title(f"HadISST SST on {selected_date}")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_xlim(float(np.nanmin(lon)), float(np.nanmax(lon)))
    ax.set_ylim(float(np.nanmin(lat)), float(np.nanmax(lat)))
    ax.grid(alpha=0.2, linewidth=0.5)
    fig.tight_layout()

    fig.savefig(output_path)
    print(f"Saved map: {output_path}")

    if args.show:
        plt.show()
    else:
        plt.close(fig)


def plot_animation(ds: xr.Dataset, args: argparse.Namespace, plt) -> None:
    from matplotlib.animation import FuncAnimation, PillowWriter

    if args.step < 1:
        raise ValueError("--step must be >= 1")
    if args.fps < 1:
        raise ValueError("--fps must be >= 1")

    sst = get_masked_sst(ds)
    subset = select_time_range(sst, args.start, args.end).isel(time=slice(None, None, args.step))

    frame_count = int(subset.sizes.get("time", 0))
    if frame_count == 0:
        raise ValueError("No frames selected. Check --start/--end/--step.")

    if args.max_frames > 0 and frame_count > args.max_frames:
        subset = subset.isel(time=slice(0, args.max_frames))
        frame_count = int(subset.sizes["time"])

    first_date = np.datetime_as_string(subset["time"].values[0], unit="D")
    last_date = np.datetime_as_string(subset["time"].values[-1], unit="D")
    output_path = (
        Path(args.output)
        if args.output
        else default_animation_output(first_date, last_date, args.step)
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lon = ds["longitude"].values
    lat = ds["latitude"].values
    lon_min, lon_max = float(np.nanmin(lon)), float(np.nanmax(lon))
    lat_min, lat_max = float(np.nanmin(lat)), float(np.nanmax(lat))

    first_frame = subset.isel(time=0).values
    fig, ax = plt.subplots(figsize=(12, 5), dpi=150)
    image = ax.imshow(
        first_frame,
        extent=[lon_min, lon_max, lat_min, lat_max],
        origin="upper",
        aspect="auto",
        cmap=args.cmap,
        vmin=args.vmin,
        vmax=args.vmax,
    )

    cbar = fig.colorbar(image, ax=ax, pad=0.02)
    cbar.set_label("SST (degC)")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(alpha=0.2, linewidth=0.5)

    def update(frame_idx: int):
        frame = subset.isel(time=frame_idx)
        frame_date = np.datetime_as_string(frame["time"].values, unit="D")
        image.set_data(frame.values)
        ax.set_title(f"HadISST SST on {frame_date}")
        return (image,)

    anim = FuncAnimation(
        fig,
        update,
        frames=frame_count,
        interval=1000 / args.fps,
        blit=True,
    )

    writer = PillowWriter(fps=args.fps)
    anim.save(output_path, writer=writer)
    print(f"Saved animation: {output_path} ({frame_count} frames)")
    plt.close(fig)


def plot_point_series(ds: xr.Dataset, args: argparse.Namespace, plt) -> None:
    if args.lon is None or args.lat is None:
        raise ValueError("series mode requires --lon and --lat")

    sst = get_masked_sst(ds)
    point = sst.sel(longitude=args.lon, latitude=args.lat, method="nearest")
    point = select_time_range(point, args.start, args.end)

    if int(point.sizes.get("time", 0)) == 0:
        raise ValueError("No series points selected. Check --start and --end.")

    matched_lon = float(point["longitude"].values)
    matched_lat = float(point["latitude"].values)
    output_path = Path(args.output) if args.output else default_series_output(matched_lon, matched_lat)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rolling_series = None
    if args.rolling > 1:
        rolling_series = point.rolling(time=args.rolling, center=True).mean()

    fig, ax = plt.subplots(figsize=(12, 4.8), dpi=150)
    ax.plot(point["time"].values, point.values, linewidth=1.0, alpha=0.65, label="Monthly SST")

    if rolling_series is not None:
        ax.plot(
            rolling_series["time"].values,
            rolling_series.values,
            linewidth=1.8,
            label=f"{args.rolling}-month rolling mean",
        )

    ax.set_title(f"SST Time Series near lon={matched_lon:.1f}, lat={matched_lat:.1f}")
    ax.set_xlabel("Time")
    ax.set_ylabel("SST (degC)")
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.legend()
    fig.tight_layout()

    fig.savefig(output_path)
    print(f"Saved series figure: {output_path}")

    csv_path = Path(args.csv_output) if args.csv_output else output_path.with_suffix(".csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    df = point.to_dataframe(name="sst").reset_index()[["time", "sst"]]
    if rolling_series is not None:
        df[f"sst_rolling_{args.rolling}m"] = rolling_series.values
    df.to_csv(csv_path, index=False)
    print(f"Saved series csv: {csv_path}")

    if args.show:
        plt.show()
    else:
        plt.close(fig)


def main() -> None:
    args = parse_args()

    if (not args.show) or args.mode == "animate":
        import matplotlib

        matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    ds = xr.open_dataset(args.file, decode_times=True)
    try:
        if args.mode == "map":
            plot_single_map(ds, args, plt)
        elif args.mode == "animate":
            plot_animation(ds, args, plt)
        elif args.mode == "series":
            plot_point_series(ds, args, plt)
    finally:
        ds.close()


if __name__ == "__main__":
    main()