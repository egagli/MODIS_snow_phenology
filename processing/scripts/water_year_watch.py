"""
Open a GitHub issue when a new (water year, hemisphere) becomes eligible.

Run monthly by .github/workflows/water_year_watch.yml. Eligibility is purely
date-based (season end + TRAILING_BUFFER_DAYS, same rule as
modis_snow_phenology.status.wy_eligible), and issues are deduplicated by
exact title across open AND closed states — so each (water year, hemisphere)
triggers exactly one reminder, ever, regardless of config or store state.
That matters for the hemisphere lag: northern WY N becomes eligible ~Jan
N+1, southern ~Jul N+1, and the southern reminder must still fire after the
northern one has been acted on (config bumped, store extended).

Stdlib only (no repo imports) so it runs on a bare runner; the buffer and
year floor come from parsing the flat config file directly.
"""

import datetime
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.parent
CONFIG_FILE = REPO_ROOT / "config" / "config_v1.txt"

# Years before this are handled by the initial processing era; the watcher
# only reminds about seasons that close after it was put in place.
FIRST_WATCHED_WY = 2026

ISSUE_TITLE = "Ready to process: WY{wy} ({hemi} hemisphere)"
ISSUE_BODY = """\
Water year {wy} for the **{hemi} hemisphere** (ended {end}) is now past its
{buffer}-day trailing buffer — MOD10A2 has enough post-season context for
bidirectional cloud filling.

Checklist:
- [ ] Confirm MOD10A2 is still flowing (CMR granule check, see README
      "Data source: NSIDC via earthaccess"):
      `curl -s "https://cmr.earthdata.nasa.gov/search/granules.json?short_name=MOD10A2&page_size=1&sort_key=-start_date" | python3 -c "import sys,json; print(json.load(sys.stdin)['feed']['entry'][0]['time_start'])"`
- [ ] If the store's `water_year` coordinate does not yet include {wy}:
      bump `WY_END` in `config/config_v1.txt` (+ local secrets copy), then
      `python processing/scripts/extend_store_water_years.py --config-file config/config_with_secrets_v1.txt`
- [ ] Dispatch **Process All Tiles** (which_tiles=missing) — only
      {hemi}-hemisphere tiles will pick up WY{wy}; the other hemisphere's
      tiles stay untouched until their own season closes.
- [ ] After the fleet finishes: rebuild the multiscale pyramid
      (`map/create_zarr_multiscales.ipynb`), regenerate the map status
      geojson, and add {wy} to `WATER_YEARS` in `map/lib/store.ts`.
- [ ] Then extend the downstream snowmelt-runoff dataset
      (global_snowmelt_runoff_onset gets its own reminder ~1 month later).

*Opened automatically by water_year_watch.yml.*
"""


def season_end(wy: int, hemisphere: str) -> datetime.date:
    return (datetime.date(wy, 9, 30) if hemisphere == "northern"
            else datetime.date(wy + 1, 3, 31))


def read_config_int(key: str, default: int) -> int:
    for line in CONFIG_FILE.read_text().splitlines():
        k, _, val = line.partition("=")
        if k.strip() == key:
            try:
                return int(val.strip())
            except ValueError:
                return default
    return default


def existing_issue_titles() -> set:
    out = subprocess.check_output(
        ["gh", "issue", "list", "--state", "all", "--limit", "500",
         "--json", "title"],
        text=True,
    )
    return {item["title"] for item in json.loads(out)}


def main():
    buffer_days = read_config_int("TRAILING_BUFFER_DAYS", 90)
    today = datetime.date.today()
    titles = existing_issue_titles()
    created = 0

    for hemi in ("northern", "southern"):
        for wy in range(FIRST_WATCHED_WY, today.year + 2):
            end = season_end(wy, hemi)
            if today < end + datetime.timedelta(days=buffer_days):
                continue
            title = ISSUE_TITLE.format(wy=wy, hemi=hemi)
            if title in titles:
                continue
            body = ISSUE_BODY.format(wy=wy, hemi=hemi, end=end, buffer=buffer_days)
            subprocess.run(
                ["gh", "issue", "create", "--title", title, "--body", body],
                check=True,
            )
            print(f"opened: {title}")
            created += 1

    print(f"done: {created} issue(s) opened")
    return 0


if __name__ == "__main__":
    sys.exit(main())
