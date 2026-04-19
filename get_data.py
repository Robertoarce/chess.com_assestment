from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from urllib.request import Request, urlopen
import pandas as pd

DEFAULT_USER_AGENT = "roberto-tool/1.0 (username: titorium; contact: roberto_arce_@hotmail.com)"
DEFAULT_TIMEOUT = 30
DEFAULT_RETRIES = 3
DEFAULT_RETRY_DELAY = 1.0
DEFAULT_TOURNAMENTS = [
    "https://api.chess.com/pub/tournament/titled-tuesday-blitz-february-10-2026-6221327",
    "https://api.chess.com/pub/tournament/titled-tuesday-blitz-march-10-2026-6277141",
]
LOSS_RESULTS = {
    "timeout",
    "resigned",
    "checkmated",
}
DRAW_RESULTS = {
    "agreed",
    "50move",
    "insufficient",
    "repetition",
    "stalemate",
    "timevsinsufficient",
}
ORDER_COLS = [
    "tournament_index",
    "round_index",
    "group_index",
    "game_index",
    "game_end_time",
]
HISTORY_GROUP_COLS = ["tournament_index", "player_uuid"]
SCOPE_FLAG_COLS = {
    "as_white": "_is_white",
    "as_black": "_is_black",
    "as_any": "_is_any",
}


def _resolve_url(url: str, base_url: str | None = None) -> str:
    return urljoin(base_url, url) if base_url else url


def _load_json(url: str, *, user_agent: str, timeout: int) -> dict[str, Any]:
    request = Request(url, headers={"User-Agent": user_agent})
    with urlopen(request, timeout=timeout) as response:
        return json.load(response)


def _should_retry(attempt: int, retries: int) -> bool:
    return attempt < retries


def _print_retry_message(message: str, *, verbose: bool) -> None:
    if verbose:
        print(message)


def fetch_json(
    url: str,
    *,
    user_agent: str = DEFAULT_USER_AGENT,
    timeout: int = DEFAULT_TIMEOUT,
    retries: int = DEFAULT_RETRIES,
    retry_delay: float = DEFAULT_RETRY_DELAY,
    base_url: str | None = None,
    raise_on_404: bool = False,
    verbose: bool = True,
) -> dict[str, Any] | None:
    resolved_url = _resolve_url(url, base_url)

    for attempt in range(1, retries + 1):
        try:
            return _load_json(resolved_url, user_agent=user_agent, timeout=timeout)
        except HTTPError as error:
            if error.code == 404 and not raise_on_404:
                _print_retry_message(
                    f"Skipping missing URL (404): {resolved_url}",
                    verbose=verbose,
                )
                return None
            if not _should_retry(attempt, retries):
                raise
            _print_retry_message(
                (
                    f"HTTP error {error.code} for {resolved_url}. "
                    f"Retrying {attempt}/{retries}..."
                ),
                verbose=verbose,
            )
        except URLError as error:
            if not _should_retry(attempt, retries):
                raise
            _print_retry_message(
                (
                    f"Network error for {resolved_url}: {error}. "
                    f"Retrying {attempt}/{retries}..."
                ),
                verbose=verbose,
            )

        time.sleep(retry_delay * attempt)

    return None


def game_to_row(
    game: dict[str, Any],
    *,
    tournament_name: str,
    tournament_index: int,
    tournament_time: str,
    tournament_date_start: pd.Timestamp,
    tournament_date_end: pd.Timestamp,
    round_index: int,
    group_index: int,
    game_index: int,
) -> dict[str, Any]:
    white_player = game["white"]
    black_player = game["black"]

    white_rating = white_player["rating"]
    black_rating = black_player["rating"]

    return {
        "tournament_name": tournament_name,
        "tournament_index": tournament_index,
        "tournament_time": tournament_time,
        "tournament_date_start": tournament_date_start,
        "tournament_date_end": tournament_date_end,
        "group_index": group_index,
        "round_index": round_index,
        "game_index": game_index,
        "game_end_time": pd.to_datetime(game["end_time"], unit="s", utc=True),
        "game_is_rated": game["rated"],
        "game_time_control": game["time_control"],
        "game_opening": game.get("eco"),
        "white_player": white_player["username"].lower(),
        "white_rating": white_rating,
        "white_result": white_player["result"],
        "white_uuid": white_player["uuid"],
        "black_player": black_player["username"].lower(),
        "black_rating": black_rating,
        "black_result": black_player["result"],
        "black_uuid": black_player["uuid"],
        "rating_diff": white_rating - black_rating,
        "rating_diff_sq": (white_rating - black_rating) ** 2,
        "rating_diff_abs": abs(white_rating - black_rating),
        "rating_ratio": white_rating / black_rating if black_rating != 0 else 1.0,
        "rating_mean": (white_rating + black_rating) / 2.0,
        "white_is_higher_rated": white_rating > black_rating,
    }


def _fetch_round_data(
    round_url: str,
    *,
    tournament_url: str,
    user_agent: str,
    timeout: int,
    retries: int,
    retry_delay: float,
    verbose: bool,
) -> dict[str, Any] | None:
    return fetch_json(
        round_url,
        user_agent=user_agent,
        timeout=timeout,
        retries=retries,
        retry_delay=retry_delay,
        base_url=tournament_url,
        verbose=verbose,
    )


def _fetch_group_data(
    group_url: str,
    *,
    round_url: str,
    user_agent: str,
    timeout: int,
    retries: int,
    retry_delay: float,
    verbose: bool,
) -> dict[str, Any] | None:
    return fetch_json(
        group_url,
        user_agent=user_agent,
        timeout=timeout,
        retries=retries,
        retry_delay=retry_delay,
        base_url=round_url,
        verbose=verbose,
    )


def _build_group_rows(
    group_data: dict[str, Any],
    *,
    tournament_name: str,
    tournament_index: int,
    tournament_time: str,
    tournament_date_start: pd.Timestamp,
    tournament_date_end: pd.Timestamp,
    round_index: int,
    group_index: int,
    verbose: bool,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for game_index, game in enumerate(group_data.get("games", []), start=1):
        try:
            rows.append(
                game_to_row(
                    game,
                    tournament_name=tournament_name,
                    tournament_index=tournament_index,
                    tournament_time=tournament_time,
                    tournament_date_start=tournament_date_start,
                    tournament_date_end=tournament_date_end,
                    round_index=round_index,
                    group_index=group_index,
                    game_index=game_index,
                )
            )
        except KeyError as error:
            if error.args and error.args[0] == "uuid":
                white_username = game.get("white", {}).get("username", "<unknown>")
                black_username = game.get("black", {}).get("username", "<unknown>")
                _print_retry_message(
                    (
                        "Skipping game due to missing uuid: "
                        f"tournament={tournament_name}, round={round_index}, "
                        f"group={group_index}, game={game_index}, "
                        f"white={white_username}, black={black_username}, error={error}"
                    ),
                    verbose=verbose,
                )
                continue
            raise

    return rows


def _build_round_rows(
    round_data: dict[str, Any],
    *,
    round_url: str,
    tournament_name: str,
    tournament_index: int,
    tournament_time: str,
    tournament_date_start: pd.Timestamp,
    tournament_date_end: pd.Timestamp,
    round_index: int,
    user_agent: str,
    timeout: int,
    retries: int,
    retry_delay: float,
    verbose: bool,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for group_index, group_url in enumerate(round_data.get("groups", []), start=1):
        group_data = _fetch_group_data(
            group_url,
            round_url=round_url,
            user_agent=user_agent,
            timeout=timeout,
            retries=retries,
            retry_delay=retry_delay,
            verbose=verbose,
        )
        if group_data is None:
            continue

        rows.extend(
            _build_group_rows(
                group_data,
                tournament_name=tournament_name,
                tournament_index=tournament_index,
                tournament_time=tournament_time,
                tournament_date_start=tournament_date_start,
                tournament_date_end=tournament_date_end,
                round_index=round_index,
                group_index=group_index,
                verbose=verbose,
            )
        )

    return rows


def _build_tournament_rows(
    tournament_data: dict[str, Any],
    *,
    tournament_url: str,
    tournament_index: int,
    user_agent: str,
    timeout: int,
    retries: int,
    retry_delay: float,
    verbose: bool,
) -> list[dict[str, Any]]:
    tournament_name = tournament_data["name"]
    tournament_time = tournament_data["settings"]["time_control"]
    tournament_date_start = pd.to_datetime(tournament_data["start_time"], unit="s", utc=True)
    tournament_date_end = pd.to_datetime(tournament_data["finish_time"], unit="s", utc=True)
    rows: list[dict[str, Any]] = []

    for round_index, round_url in enumerate(tournament_data["rounds"], start=1):
        round_data = _fetch_round_data(
            round_url,
            tournament_url=tournament_url,
            user_agent=user_agent,
            timeout=timeout,
            retries=retries,
            retry_delay=retry_delay,
            verbose=verbose,
        )
        if round_data is None:
            continue

        rows.extend(
            _build_round_rows(
                round_data,
                round_url=round_url,
                tournament_name=tournament_name,
                tournament_index=tournament_index,
                tournament_time=tournament_time,
                tournament_date_start=tournament_date_start,
                tournament_date_end=tournament_date_end,
                round_index=round_index,
                user_agent=user_agent,
                timeout=timeout,
                retries=retries,
                retry_delay=retry_delay,
                verbose=verbose,
            )
        )

    return rows


def build_games_dataframe(
    tournaments: list[str] | None = None,
    *,
    user_agent: str = DEFAULT_USER_AGENT,
    timeout: int = DEFAULT_TIMEOUT,
    retries: int = DEFAULT_RETRIES,
    retry_delay: float = DEFAULT_RETRY_DELAY,
    verbose: bool = True,
) -> pd.DataFrame:
    tournament_urls = tournaments or DEFAULT_TOURNAMENTS
    rows: list[dict[str, Any]] = []

    for tournament_index, tournament_url in enumerate(tournament_urls, start=1):
        tournament_data = fetch_json(
            tournament_url,
            user_agent=user_agent,
            timeout=timeout,
            retries=retries,
            retry_delay=retry_delay,
            verbose=verbose,
        )
        if tournament_data is None:
            continue

        rows.extend(
            _build_tournament_rows(
                tournament_data,
                tournament_url=tournament_url,
                tournament_index=tournament_index,
                user_agent=user_agent,
                timeout=timeout,
                retries=retries,
                retry_delay=retry_delay,
                verbose=verbose,
            )
        )

    return pd.DataFrame(rows)


def save_games_csv(
    output_path: str | Path = "titled_tuesday_games.csv",
    tournaments: list[str] | None = None,
    *,
    user_agent: str = DEFAULT_USER_AGENT,
    timeout: int = DEFAULT_TIMEOUT,
    retries: int = DEFAULT_RETRIES,
    retry_delay: float = DEFAULT_RETRY_DELAY,
    verbose: bool = True,
) -> pd.DataFrame:
    dataframe = build_games_dataframe(
        tournaments=tournaments,
        user_agent=user_agent,
        timeout=timeout,
        retries=retries,
        retry_delay=retry_delay,
        verbose=verbose,
    )
    path = Path(output_path)
    dataframe.to_csv(path, index=False)
    if verbose:
        print(f"Saved {len(dataframe)} rows to {path}")
    return dataframe


def white_outcome_from_result(result: str) -> str:
    if result in LOSS_RESULTS:
        return "loss"
    if result in DRAW_RESULTS:
        return "draw"
    return "win"


def points_from_outcome(outcome: str) -> float:
    if outcome == "win":
        return 1.0
    if outcome == "draw":
        return 0.5
    return 0.0


def _build_player_appearances(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = df.sort_values(ORDER_COLS).copy()
    out["_row_id"] = out.index

    base_cols = ORDER_COLS + ["_row_id"]

    white = out[base_cols + ["white_uuid", "white_outcome", "white_points"]].copy()
    white.columns = base_cols + ["player_uuid", "outcome", "points"]
    white["row_prefix"] = "white"
    white["appearance_side"] = "white"

    black = out[base_cols + ["black_uuid", "black_outcome", "black_points"]].copy()
    black.columns = base_cols + ["player_uuid", "outcome", "points"]
    black["row_prefix"] = "black"
    black["appearance_side"] = "black"

    appearances = pd.concat([white, black], ignore_index=True)
    appearances = appearances.sort_values(ORDER_COLS + ["_row_id", "row_prefix"]).reset_index(drop=True)

    appearances["_is_white"] = (appearances["appearance_side"] == "white").astype(int)
    appearances["_is_black"] = (appearances["appearance_side"] == "black").astype(int)
    appearances["_is_any"] = 1

    appearances["_is_win"] = (appearances["outcome"] == "win").astype(int)
    appearances["_is_draw"] = (appearances["outcome"] == "draw").astype(int)
    appearances["_is_loss"] = (appearances["outcome"] == "loss").astype(int)

    return out, appearances


def _add_scope_history_features(appearances: pd.DataFrame, scope_label: str) -> pd.DataFrame:
    scope_flag_col = SCOPE_FLAG_COLS[scope_label]
    grouped = appearances.groupby(HISTORY_GROUP_COLS, sort=False)

    appearances[f"_points_{scope_label}"] = appearances["points"] * appearances[scope_flag_col]
    appearances[f"_wins_{scope_label}"] = appearances["_is_win"] * appearances[scope_flag_col]
    appearances[f"_draws_{scope_label}"] = appearances["_is_draw"] * appearances[scope_flag_col]
    appearances[f"_losses_{scope_label}"] = appearances["_is_loss"] * appearances[scope_flag_col]

    previous_games_col = f"previous_games_{scope_label}"
    appearances[previous_games_col] = (
        grouped[scope_flag_col].cumsum() - appearances[scope_flag_col]
    ).astype(int)

    appearances[f"previous_points_{scope_label}"] = (
        grouped[f"_points_{scope_label}"].cumsum() - appearances[f"_points_{scope_label}"]
    )

    previous_wins = (
        grouped[f"_wins_{scope_label}"].cumsum() - appearances[f"_wins_{scope_label}"]
    ).astype(int)

    appearances[f"previous_draws_{scope_label}"] = (
        grouped[f"_draws_{scope_label}"].cumsum() - appearances[f"_draws_{scope_label}"]
    ).astype(int)

    appearances[f"previous_losses_{scope_label}"] = (
        grouped[f"_losses_{scope_label}"].cumsum() - appearances[f"_losses_{scope_label}"]
    ).astype(int)

    previous_games = appearances[previous_games_col].where(
        appearances[previous_games_col] > 0
    )

    appearances[f"previous_win_pct_{scope_label}"] = (
        previous_wins.div(previous_games).fillna(0.0) * 100.0
    )
    appearances[f"previous_draw_pct_{scope_label}"] = (
        appearances[f"previous_draws_{scope_label}"].div(previous_games).fillna(0.0) * 100.0
    )
    appearances[f"previous_loss_pct_{scope_label}"] = (
        appearances[f"previous_losses_{scope_label}"].div(previous_games).fillna(0.0) * 100.0
    )

    return appearances


def add_player_history_features(
    df: pd.DataFrame,
    scopes: tuple[str, ...] = ("as_white", "as_black", "as_any"),
) -> pd.DataFrame:
    out, appearances = _build_player_appearances(df)
    feature_cols: list[str] = []

    for scope_label in scopes:
        appearances = _add_scope_history_features(appearances, scope_label)
        scoped_feature_cols = [
            f"previous_points_{scope_label}",
            f"previous_draws_{scope_label}",
            f"previous_losses_{scope_label}",
            f"previous_win_pct_{scope_label}",
            f"previous_draw_pct_{scope_label}",
            f"previous_loss_pct_{scope_label}",
        ]
        if scope_label != "as_any":
            scoped_feature_cols.insert(0, f"previous_games_{scope_label}")
        feature_cols.extend(scoped_feature_cols)

    for row_prefix in ("white", "black"):
        row_features = appearances[appearances["row_prefix"] == row_prefix].set_index("_row_id")

        for feature_col in feature_cols:
            new_col = f"{row_prefix}_{feature_col}"
            values = row_features[feature_col].reindex(out.index).fillna(0)

            if feature_col.startswith("previous_points_") or "_pct_" in feature_col:
                out[new_col] = values.astype(float)
            else:
                out[new_col] = values.astype(int)

    return out.drop(columns=["_row_id"]).sort_index()


def engineer_games_dataframe(dataframe: pd.DataFrame) -> pd.DataFrame:
    df = dataframe.copy()
    if df.empty:
        return df

    df["white_outcome"] = df["white_result"].apply(white_outcome_from_result)
    df["black_outcome"] = df["white_outcome"].map({"win": "loss", "loss": "win", "draw": "draw"})
    df["white_points"] = df["white_outcome"].apply(points_from_outcome)
    df["black_points"] = df["black_outcome"].apply(points_from_outcome)

    return add_player_history_features(df)


def build_finished_games_dataframe(
    tournaments: list[str] | None = None,
    *,
    raw_dataframe: pd.DataFrame | None = None,
    user_agent: str = DEFAULT_USER_AGENT,
    timeout: int = DEFAULT_TIMEOUT,
    retries: int = DEFAULT_RETRIES,
    retry_delay: float = DEFAULT_RETRY_DELAY,
    verbose: bool = True,
) -> pd.DataFrame:
    source_dataframe = raw_dataframe
    if source_dataframe is None:
        print(" Fetching raw games data from API...")
        source_dataframe = build_games_dataframe(
            tournaments=tournaments,
            user_agent=user_agent,
            timeout=timeout,
            retries=retries,
            retry_delay=retry_delay,
            verbose=verbose,
        )

    return engineer_games_dataframe(source_dataframe)


def save_finished_games_csv(
    output_path: str | Path = "titled_tuesday_games_finished.csv",
    tournaments: list[str] | None = None,
    *,
    raw_dataframe: pd.DataFrame | None = None,
    user_agent: str = DEFAULT_USER_AGENT,
    timeout: int = DEFAULT_TIMEOUT,
    retries: int = DEFAULT_RETRIES,
    retry_delay: float = DEFAULT_RETRY_DELAY,
    verbose: bool = True,
) -> pd.DataFrame:
    print(" Saving finished dataframe into CSV...")
    dataframe = build_finished_games_dataframe(
        tournaments=tournaments,
        raw_dataframe=raw_dataframe,
        user_agent=user_agent,
        timeout=timeout,
        retries=retries,
        retry_delay=retry_delay,
        verbose=verbose,
    )
    path = Path(output_path)
    dataframe.to_csv(path, index=False)
    if verbose:
        print(f"Saved {len(dataframe)} rows to {path}")
    return dataframe


if __name__ == "__main__":
    raw_df = save_games_csv()
    save_finished_games_csv(raw_dataframe=raw_df)