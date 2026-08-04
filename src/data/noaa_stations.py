"""NOAA NCEI ISD station definitions for top 10 US MSAs (2020 Census).

Reference papers using this aggregation approach:
- Shaman, J. et al. (2013). "Real-time influenza forecasts during the 2012-2013
  season." Nature Communications.
- Reich, N.G. et al. (2019). "A collaborative multiyear, multimodel assessment
  of seasonal influenza forecasting in the United States." PNAS.

Station selection:
- 1 primary airport ASOS station per MSA (highest data completeness)
- USAF + WBAN 11-digit ISD ID (concatenation, no dash)
- 2020 Census MSA population (US Census Bureau Decennial Census)

Note: Population weights are normalized so that Σ weights == 1.0 (computed
on import). Use `STATION_WEIGHTS_DICT[isd_id]` for the normalized weight.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MSAStation:
    """One primary ASOS station for a top US MSA."""
    rank: int                  # MSA rank by 2020 population
    msa_name: str              # Full Metropolitan Statistical Area name
    short_name: str            # Short display name
    state: str                 # Primary state (for tie-break)
    airport_iata: str          # e.g., 'LGA'
    isd_id: str                # 11-digit USAF + WBAN concatenated
    usaf: str                  # 6-digit USAF
    wban: str                  # 5-digit WBAN
    population_2020: int       # MSA population from 2020 Census


# Top 10 US MSAs (2020 Census Bureau Decennial Census, ranked by population)
# Source: https://www.census.gov/data/tables/time-series/demo/popest/2020s-total-metro-and-micro-statistical-areas.html
MSA_STATIONS: list[MSAStation] = [
    MSAStation(
        rank=1,
        msa_name="New York-Newark-Jersey City",
        short_name="New York",
        state="NY/NJ/PA",
        airport_iata="LGA",
        isd_id="72503014732",
        usaf="725030",
        wban="14732",
        population_2020=20_140_470,
    ),
    MSAStation(
        rank=2,
        msa_name="Los Angeles-Long Beach-Anaheim",
        short_name="Los Angeles",
        state="CA",
        airport_iata="LAX",
        isd_id="72295023174",
        usaf="722950",
        wban="23174",
        population_2020=13_200_998,
    ),
    MSAStation(
        rank=3,
        msa_name="Chicago-Naperville-Elgin",
        short_name="Chicago",
        state="IL/IN/WI",
        airport_iata="ORD",
        isd_id="72530094846",
        usaf="725300",
        wban="94846",
        population_2020=9_618_502,
    ),
    MSAStation(
        rank=4,
        msa_name="Dallas-Fort Worth-Arlington",
        short_name="Dallas",
        state="TX",
        airport_iata="DFW",
        isd_id="72259003927",
        usaf="722590",
        wban="03927",
        population_2020=7_637_387,
    ),
    MSAStation(
        rank=5,
        msa_name="Houston-The Woodlands-Sugar Land",
        short_name="Houston",
        state="TX",
        airport_iata="IAH",
        isd_id="72243012960",
        usaf="722430",
        wban="12960",
        population_2020=7_122_240,
    ),
    MSAStation(
        rank=6,
        msa_name="Washington-Arlington-Alexandria",
        short_name="Washington",
        state="DC/VA/MD/WV",
        airport_iata="DCA",
        isd_id="72405013743",
        usaf="724050",
        wban="13743",
        population_2020=6_385_162,
    ),
    MSAStation(
        rank=7,
        msa_name="Philadelphia-Camden-Wilmington",
        short_name="Philadelphia",
        state="PA/NJ/DE/MD",
        airport_iata="PHL",
        isd_id="72408013739",
        usaf="724080",
        wban="13739",
        population_2020=6_245_051,
    ),
    MSAStation(
        rank=8,
        msa_name="Miami-Fort Lauderdale-Pompano Beach",
        short_name="Miami",
        state="FL",
        airport_iata="MIA",
        isd_id="72202012839",
        usaf="722020",
        wban="12839",
        population_2020=6_138_333,
    ),
    MSAStation(
        rank=9,
        msa_name="Atlanta-Sandy Springs-Alpharetta",
        short_name="Atlanta",
        state="GA",
        airport_iata="ATL",
        isd_id="72219013874",
        usaf="722190",
        wban="13874",
        population_2020=6_089_815,
    ),
    MSAStation(
        rank=10,
        msa_name="Phoenix-Mesa-Chandler",
        short_name="Phoenix",
        state="AZ",
        airport_iata="PHX",
        isd_id="72278023183",
        usaf="722780",
        wban="23183",
        population_2020=4_845_832,
    ),
]


# Total 2020 population across top 10 MSAs
TOTAL_POPULATION_2020: int = sum(s.population_2020 for s in MSA_STATIONS)


def get_normalized_weights() -> dict[str, float]:
    """Return {isd_id: normalized_weight} where Σ weights == 1.0."""
    return {s.isd_id: s.population_2020 / TOTAL_POPULATION_2020 for s in MSA_STATIONS}


# Pre-computed normalized weights dict (loaded on import)
STATION_WEIGHTS_DICT: dict[str, float] = get_normalized_weights()


def get_isd_url(isd_id: str, year: int) -> str:
    """Return the NOAA NCEI ISD bulk download URL for one station-year.

    Example:
        >>> get_isd_url("72503014732", 2020)
        'https://www.ncei.noaa.gov/data/global-hourly/access/2020/72503014732.csv'
    """
    return f"https://www.ncei.noaa.gov/data/global-hourly/access/{year}/{isd_id}.csv"


if __name__ == "__main__":
    # Print station roster + weights as a summary table
    print(f"{'Rank':>4} {'Short':<14} {'IATA':<5} {'ISD ID':<13} "
          f"{'Pop 2020':>13} {'Weight':>8}")
    print("-" * 70)
    for s in MSA_STATIONS:
        w = STATION_WEIGHTS_DICT[s.isd_id]
        print(f"{s.rank:>4} {s.short_name:<14} {s.airport_iata:<5} "
              f"{s.isd_id:<13} {s.population_2020:>13,} {w:>8.4f}")
    print("-" * 70)
    print(f"{'Total':>4} {'':<14} {'':<5} {'':<13} "
          f"{TOTAL_POPULATION_2020:>13,} {sum(STATION_WEIGHTS_DICT.values()):>8.4f}")
    print()
    print("Sample URL (LGA, year 2020):")
    print(f"  {get_isd_url('72503014732', 2020)}")
