"""
LSTM-Transformer Dataset
Creates sliding-window time series samples for PyTorch training.
Each sample: [30 days of history + context] -> [16 days of future weather]
"""
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from lstm_transformer import config


class WeatherSequenceDataset(Dataset):
    """
    PyTorch Dataset for weather time series forecasting.

    Each sample contains:
    - weather_seq: (seq_length, n_weather_features) — 30 days of local weather
    - context_seq: (seq_length, n_context_features) — 30 days of India-wide context
    - targets: (forecast_horizon, n_targets) — 16 days of future weather
    - target_mask: (forecast_horizon, n_targets) — 1 where target is valid, 0 for NaN
    """

    def __init__(self, weather_df, context_df=None, city_list=None,
                 seq_length=None, forecast_horizon=None, feature_stats=None):
        """
        Args:
            weather_df: DataFrame with Date, City, and weather columns
            context_df: DataFrame with Date and context features (optional)
            city_list: List of cities to include (None = all)
            seq_length: Number of past days (default from config)
            forecast_horizon: Number of future days (default from config)
            feature_stats: Dictionary of means/stds for scaling (None = compute from data)
        """
        self.seq_length = seq_length or config.SEQ_LENGTH
        self.forecast_horizon = forecast_horizon or config.FORECAST_HORIZON
        self.target_names = config.TARGET_COLS
        self.feature_stats = feature_stats

        # Identify feature columns
        self.weather_feature_cols = [c for c in config.WEATHER_FEATURE_COLS
                                      if c in weather_df.columns]

        # Wind encoding
        if config.WIND_DIR_COL in weather_df.columns:
            weather_df = weather_df.copy()
            wind_rad = np.deg2rad(weather_df[config.WIND_DIR_COL].fillna(0))
            weather_df["Wind_Dir_Sin"] = np.sin(wind_rad)
            weather_df["Wind_Dir_Cos"] = np.cos(wind_rad)
            self.weather_feature_cols += ["Wind_Dir_Sin", "Wind_Dir_Cos"]

        # Filter cities
        if city_list is not None:
            weather_df = weather_df[weather_df["City"].isin(city_list)].copy()

        weather_df = weather_df.sort_values(["City", "Date"]).reset_index(drop=True)

        # Build per-city sequences
        self.samples = []
        self._build_samples(weather_df, context_df)

        print(f"  Dataset: {len(self.samples)} samples, "
              f"seq_len={self.seq_length}, horizon={self.forecast_horizon}, "
              f"weather_feats={len(self.weather_feature_cols)}")

    def _build_samples(self, weather_df, context_df):
        """Build all (input_seq, target) pairs from the data."""

        # Prepare context features if available
        context_features = None
        context_cols = []
        if context_df is not None and not context_df.empty:
            context_df = context_df.sort_values("Date").reset_index(drop=True)
            # Only keep numeric columns and Date
            numeric_cols = context_df.select_dtypes(include=[np.number]).columns.tolist()
            context_cols = [c for c in numeric_cols if c != "Date"]
            context_df = context_df.set_index("Date")
            context_features = context_df

        self.n_weather_features = len(self.weather_feature_cols)
        self.n_context_features = len(context_cols)
        self.n_targets = len(self.target_names)

        all_weather_vals = []
        all_ctx_vals = []
        city_data_blocks = []

        for city, city_df in weather_df.groupby("City"):
            city_df = city_df.sort_values("Date").reset_index(drop=True)
            dates = city_df["Date"].values

            # Extract weather features as numpy array
            weather_vals = city_df[self.weather_feature_cols].values.astype(np.float32)

            # Fill NaN with column medians
            for col_idx in range(weather_vals.shape[1]):
                col_data = weather_vals[:, col_idx]
                nan_mask = np.isnan(col_data)
                if nan_mask.any():
                    median_val = np.nanmedian(col_data)
                    if np.isnan(median_val):
                        median_val = 0.0
                    weather_vals[nan_mask, col_idx] = median_val

            # Extract target values
            target_cols_avail = [c for c in self.target_names if c in city_df.columns]
            target_vals = city_df[target_cols_avail].values.astype(np.float32)

            # Log-transform precipitation
            if "Precipitation_mm" in target_cols_avail:
                precip_idx = target_cols_avail.index("Precipitation_mm")
                target_vals[:, precip_idx] = np.log1p(np.maximum(target_vals[:, precip_idx], 0))
            if "Precipitation_mm" in self.weather_feature_cols:
                feat_precip_idx = self.weather_feature_cols.index("Precipitation_mm")
                weather_vals[:, feat_precip_idx] = np.log1p(np.maximum(weather_vals[:, feat_precip_idx], 0))

            n_days = len(city_df)
            ctx_array = None
            if context_features is not None and len(context_cols) > 0:
                ctx_array = np.zeros((n_days, len(context_cols)), dtype=np.float32)
                for i, d in enumerate(dates):
                    d_ts = pd.Timestamp(d)
                    if d_ts in context_features.index:
                        ctx_array[i] = np.nan_to_num(context_features.loc[d_ts, context_cols].values.astype(np.float32), nan=0.0)
            else:
                ctx_array = np.zeros((n_days, 1), dtype=np.float32)

            all_weather_vals.append(weather_vals)
            all_ctx_vals.append(ctx_array)
            city_data_blocks.append((city, dates, weather_vals, ctx_array, target_vals, n_days))

        # Compute global feature stats for scaling if not provided
        if self.feature_stats is None:
            stacked_weather = np.vstack(all_weather_vals)
            stacked_ctx = np.vstack(all_ctx_vals)
            self.feature_stats = {
                "weather_mean": np.nanmean(stacked_weather, axis=0),
                "weather_std": np.nanstd(stacked_weather, axis=0) + 1e-6,
                "context_mean": np.nanmean(stacked_ctx, axis=0),
                "context_std": np.nanstd(stacked_ctx, axis=0) + 1e-6,
            }

        # Build sliding windows
        total_window = self.seq_length + self.forecast_horizon
        for city, dates, weather_vals, ctx_array, target_vals, n_days in city_data_blocks:
            
            # Apply standard scaling to features
            scaled_weather_vals = (weather_vals - self.feature_stats["weather_mean"]) / self.feature_stats["weather_std"]
            scaled_ctx_array = (ctx_array - self.feature_stats["context_mean"]) / self.feature_stats["context_std"]

            for start_idx in range(0, n_days - total_window + 1, 1):
                end_input = start_idx + self.seq_length
                end_target = end_input + self.forecast_horizon

                weather_seq = scaled_weather_vals[start_idx:end_input]
                context_seq = scaled_ctx_array[start_idx:end_input]
                target_seq = target_vals[end_input:end_target]
                
                target_mask = (~np.isnan(target_seq)).astype(np.float32)
                target_seq = np.nan_to_num(target_seq, nan=0.0)

                self.samples.append({
                    "weather_seq": weather_seq,
                    "context_seq": context_seq,
                    "targets": target_seq,
                    "target_mask": target_mask,
                    "date": dates[end_input - 1],
                    "city": city,
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return {
            "weather_seq": torch.tensor(sample["weather_seq"], dtype=torch.float32),
            "context_seq": torch.tensor(sample["context_seq"], dtype=torch.float32),
            "targets": torch.tensor(sample["targets"], dtype=torch.float32),
            "target_mask": torch.tensor(sample["target_mask"], dtype=torch.float32),
        }

    def get_feature_dims(self):
        """Return input dimensions for model construction."""
        return {
            "n_weather_features": self.n_weather_features,
            "n_context_features": self.n_context_features,
            "n_targets": self.n_targets,
            "weather_mean": self.feature_stats["weather_mean"],
            "weather_std": self.feature_stats["weather_std"],
            "context_mean": self.feature_stats["context_mean"],
            "context_std": self.feature_stats["context_std"],
        }


def load_and_prepare_data(test_mode=False):
    """
    Load all datasets and prepare them for the LSTM-Transformer model.
    """
    print("Loading datasets for LSTM-Transformer...")

    # Load pre-merged historical data
    weather_df = pd.read_csv(config.HISTORICAL_CSV)
    weather_df["Date"] = pd.to_datetime(weather_df["Date"])
    print(f"  Historical data (merged): {len(weather_df)} rows")

    # Compute days since satellite update
    # Scene_ID changes when a new satellite image is used
    if "Scene_ID" in weather_df.columns:
        weather_df = weather_df.sort_values(["City", "Date"])
        # A new image is when Scene_ID is different from the previous row
        is_new_image = weather_df["Scene_ID"] != weather_df.groupby("City")["Scene_ID"].shift(1)
        # Also the first row for each city is a "new image" if Scene_ID is not null
        is_new_image = is_new_image & weather_df["Scene_ID"].notna()
        
        weather_df["last_update_date"] = weather_df["Date"].where(is_new_image).groupby(weather_df["City"]).ffill()
        weather_df["days_since_satellite_update"] = (weather_df["Date"] - weather_df["last_update_date"]).dt.days.fillna(0)
        weather_df = weather_df.drop(columns=["last_update_date"])
        print(f"  Computed days_since_satellite_update")
    else:
        weather_df["days_since_satellite_update"] = 0.0

    # Load pre-built context features and aggregate across sentinel cities
    context_df = pd.DataFrame()
    context_csv_path = os.path.join(os.path.dirname(config.HISTORICAL_CSV), "ml_ready_context_data.csv")
    try:
        if os.path.exists(context_csv_path):
            raw_ctx = pd.read_csv(context_csv_path)
            raw_ctx["Date"] = pd.to_datetime(raw_ctx["Date"])
            # Aggregate across all cities by Date
            agg_cols = raw_ctx.select_dtypes(include=[np.number]).columns.tolist()
            agg_cols = [c for c in agg_cols if c not in ["Year", "Month", "Day"]]
            context_df = raw_ctx.groupby("Date")[agg_cols].mean().reset_index()
            print(f"  Context features (aggregated): {len(context_df.columns) - 1} signals")
        else:
            print(f"  Context data not found at {context_csv_path}")
    except Exception as e:
        print(f"  Context building skipped: {e}")

    # Get city list
    all_cities = sorted(weather_df["City"].unique())
    if test_mode:
        city_list = all_cities[:3]
        print(f"  TEST MODE: Using {len(city_list)} cities: {city_list}")
    else:
        city_list = all_cities
        print(f"  Using all {len(city_list)} cities")

    return weather_df, context_df, city_list


def create_dataloaders(test_mode=False, batch_size=None):
    """
    Create train/val/test DataLoaders.

    Returns:
        train_loader, val_loader, test_loader, feature_dims
    """
    import os
    batch_size = batch_size or config.BATCH_SIZE

    weather_df, context_df, city_list = load_and_prepare_data(test_mode=test_mode)

    # Add Year column for splitting
    weather_df["Year"] = weather_df["Date"].dt.year

    # Split by year
    train_df = weather_df[weather_df["Year"] <= config.TRAIN_END_YEAR]
    val_df = weather_df[(weather_df["Year"] > config.TRAIN_END_YEAR) &
                         (weather_df["Year"] <= config.VAL_END_YEAR)]
    test_df = weather_df[weather_df["Year"] > config.VAL_END_YEAR]

    print(f"\n  Train: {len(train_df)} rows ({train_df['Year'].min()}-{train_df['Year'].max()})")
    print(f"  Val:   {len(val_df)} rows ({val_df['Year'].min()}-{val_df['Year'].max()})")
    print(f"  Test:  {len(test_df)} rows ({test_df['Year'].min()}-{test_df['Year'].max()})")

    # Create datasets
    print("\nBuilding training sequences...")
    train_dataset = WeatherSequenceDataset(
        train_df, context_df, city_list,
        seq_length=config.SEQ_LENGTH,
        forecast_horizon=config.FORECAST_HORIZON,
    )

    print("Building validation sequences...")
    val_dataset = WeatherSequenceDataset(
        val_df, context_df, city_list,
        seq_length=config.SEQ_LENGTH,
        forecast_horizon=config.FORECAST_HORIZON,
        feature_stats=train_dataset.feature_stats,
    )

    print("Building test sequences...")
    test_dataset = WeatherSequenceDataset(
        test_df, context_df, city_list,
        seq_length=config.SEQ_LENGTH,
        forecast_horizon=config.FORECAST_HORIZON,
        feature_stats=train_dataset.feature_stats,
    )

    # Create DataLoaders with multiprocessing to feed the GPU faster
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True, persistent_workers=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True, persistent_workers=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True, persistent_workers=True
    )

    feature_dims = train_dataset.get_feature_dims()
    return train_loader, val_loader, test_loader, feature_dims
