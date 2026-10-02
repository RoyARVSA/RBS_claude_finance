"""gics_taxonomy.py – GICS 四層分類表（working table, structure effective 2023-03；移植自使用者 gics_nn 專案）。

L1 Sector (2 位) → L2 Industry Group (4 位) → L3 Industry (6 位) → L4 Sub-Industry (8 位)
每一層的代碼都是上一層代碼的前綴，因此只要知道 L4 代碼就能唯一推回 L1–L3。
This is a working table, not official GICS Direct data.
"""
from __future__ import annotations

import difflib
import re

_RAW = """
10|Energy
1010|Energy
101010|Energy Equipment & Services
10101010|Oil & Gas Drilling
10101020|Oil & Gas Equipment & Services
101020|Oil, Gas & Consumable Fuels
10102010|Integrated Oil & Gas
10102020|Oil & Gas Exploration & Production
10102030|Oil & Gas Refining & Marketing
10102040|Oil & Gas Storage & Transportation
10102050|Coal & Consumable Fuels
15|Materials
1510|Materials
151010|Chemicals
15101010|Commodity Chemicals
15101020|Diversified Chemicals
15101030|Fertilizers & Agricultural Chemicals
15101040|Industrial Gases
15101050|Specialty Chemicals
151020|Construction Materials
15102010|Construction Materials
151030|Containers & Packaging
15103010|Metal, Glass & Plastic Containers
15103020|Paper & Plastic Packaging Products & Materials
151040|Metals & Mining
15104010|Aluminum
15104020|Diversified Metals & Mining
15104025|Copper
15104030|Gold
15104040|Precious Metals & Minerals
15104045|Silver
15104050|Steel
151050|Paper & Forest Products
15105010|Forest Products
15105020|Paper Products
20|Industrials
2010|Capital Goods
201010|Aerospace & Defense
20101010|Aerospace & Defense
201020|Building Products
20102010|Building Products
201030|Construction & Engineering
20103010|Construction & Engineering
201040|Electrical Equipment
20104010|Electrical Components & Equipment
20104020|Heavy Electrical Equipment
201050|Industrial Conglomerates
20105010|Industrial Conglomerates
201060|Machinery
20106010|Construction Machinery & Heavy Transportation Equipment
20106015|Agricultural & Farm Machinery
20106020|Industrial Machinery & Supplies & Components
201070|Trading Companies & Distributors
20107010|Trading Companies & Distributors
2020|Commercial & Professional Services
202010|Commercial Services & Supplies
20201010|Commercial Printing
20201050|Environmental & Facilities Services
20201060|Office Services & Supplies
20201070|Diversified Support Services
20201080|Security & Alarm Services
202020|Professional Services
20202010|Human Resource & Employment Services
20202020|Research & Consulting Services
20202030|Data Processing & Outsourced Services
2030|Transportation
203010|Air Freight & Logistics
20301010|Air Freight & Logistics
203020|Passenger Airlines
20302010|Passenger Airlines
203030|Marine Transportation
20303010|Marine Transportation
203040|Ground Transportation
20304010|Rail Transportation
20304030|Cargo Ground Transportation
20304040|Passenger Ground Transportation
203050|Transportation Infrastructure
20305010|Airport Services
20305020|Highways & Railtracks
20305030|Marine Ports & Services
25|Consumer Discretionary
2510|Automobiles & Components
251010|Automobile Components
25101010|Automotive Parts & Equipment
25101020|Tires & Rubber
251020|Automobiles
25102010|Automobile Manufacturers
25102020|Motorcycle Manufacturers
2520|Consumer Durables & Apparel
252010|Household Durables
25201010|Consumer Electronics
25201020|Home Furnishings
25201030|Homebuilding
25201040|Household Appliances
25201050|Housewares & Specialties
252020|Leisure Products
25202010|Leisure Products
252030|Textiles, Apparel & Luxury Goods
25203010|Apparel, Accessories & Luxury Goods
25203020|Footwear
25203030|Textiles
2530|Consumer Services
253010|Hotels, Restaurants & Leisure
25301010|Casinos & Gaming
25301020|Hotels, Resorts & Cruise Lines
25301030|Leisure Facilities
25301040|Restaurants
253020|Diversified Consumer Services
25302010|Education Services
25302020|Specialized Consumer Services
2550|Consumer Discretionary Distribution & Retail
255010|Distributors
25501010|Distributors
255030|Broadline Retail
25503030|Broadline Retail
255040|Specialty Retail
25504010|Apparel Retail
25504020|Computer & Electronics Retail
25504030|Home Improvement Retail
25504040|Other Specialty Retail
25504050|Automotive Retail
25504060|Homefurnishing Retail
30|Consumer Staples
3010|Consumer Staples Distribution & Retail
301010|Consumer Staples Distribution & Retail
30101010|Drug Retail
30101020|Food Distributors
30101030|Food Retail
30101040|Consumer Staples Merchandise Retail
3020|Food, Beverage & Tobacco
302010|Beverages
30201010|Brewers
30201020|Distillers & Vintners
30201030|Soft Drinks & Non-alcoholic Beverages
302020|Food Products
30202010|Agricultural Products & Services
30202030|Packaged Foods & Meats
302030|Tobacco
30203010|Tobacco
3030|Household & Personal Products
303010|Household Products
30301010|Household Products
303020|Personal Care Products
30302010|Personal Care Products
35|Health Care
3510|Health Care Equipment & Services
351010|Health Care Equipment & Supplies
35101010|Health Care Equipment
35101020|Health Care Supplies
351020|Health Care Providers & Services
35102010|Health Care Distributors
35102015|Health Care Services
35102020|Health Care Facilities
35102030|Managed Health Care
351030|Health Care Technology
35103010|Health Care Technology
3520|Pharmaceuticals, Biotechnology & Life Sciences
352010|Biotechnology
35201010|Biotechnology
352020|Pharmaceuticals
35202010|Pharmaceuticals
352030|Life Sciences Tools & Services
35203010|Life Sciences Tools & Services
40|Financials
4010|Banks
401010|Banks
40101010|Diversified Banks
40101015|Regional Banks
4020|Financial Services
402010|Financial Services
40201020|Diversified Financial Services
40201030|Multi-Sector Holdings
40201040|Specialized Finance
40201050|Commercial & Residential Mortgage Finance
40201060|Transaction & Payment Processing Services
402020|Consumer Finance
40202010|Consumer Finance
402030|Capital Markets
40203010|Asset Management & Custody Banks
40203020|Investment Banking & Brokerage
40203030|Diversified Capital Markets
40203040|Financial Exchanges & Data
402040|Mortgage Real Estate Investment Trusts (REITs)
40204010|Mortgage REITs
4030|Insurance
403010|Insurance
40301010|Insurance Brokers
40301020|Life & Health Insurance
40301030|Multi-line Insurance
40301040|Property & Casualty Insurance
40301050|Reinsurance
45|Information Technology
4510|Software & Services
451020|IT Services
45102010|IT Consulting & Other Services
45102030|Internet Services & Infrastructure
451030|Software
45103010|Application Software
45103020|Systems Software
4520|Technology Hardware & Equipment
452010|Communications Equipment
45201020|Communications Equipment
452020|Technology Hardware, Storage & Peripherals
45202030|Technology Hardware, Storage & Peripherals
452030|Electronic Equipment, Instruments & Components
45203010|Electronic Equipment & Instruments
45203015|Electronic Components
45203020|Electronic Manufacturing Services
45203030|Technology Distributors
4530|Semiconductors & Semiconductor Equipment
453010|Semiconductors & Semiconductor Equipment
45301010|Semiconductor Materials & Equipment
45301020|Semiconductors
50|Communication Services
5010|Telecommunication Services
501010|Diversified Telecommunication Services
50101010|Alternative Carriers
50101020|Integrated Telecommunication Services
501020|Wireless Telecommunication Services
50102010|Wireless Telecommunication Services
5020|Media & Entertainment
502010|Media
50201010|Advertising
50201020|Broadcasting
50201030|Cable & Satellite
50201040|Publishing
502020|Entertainment
50202010|Movies & Entertainment
50202020|Interactive Home Entertainment
502030|Interactive Media & Services
50203010|Interactive Media & Services
55|Utilities
5510|Utilities
551010|Electric Utilities
55101010|Electric Utilities
551020|Gas Utilities
55102010|Gas Utilities
551030|Multi-Utilities
55103010|Multi-Utilities
551040|Water Utilities
55104010|Water Utilities
551050|Independent Power and Renewable Electricity Producers
55105010|Independent Power Producers & Energy Traders
55105020|Renewable Electricity
60|Real Estate
6010|Equity Real Estate Investment Trusts (REITs)
601010|Diversified REITs
60101010|Diversified REITs
601025|Industrial REITs
60102510|Industrial REITs
601030|Hotel & Resort REITs
60103010|Hotel & Resort REITs
601040|Office REITs
60104010|Office REITs
601050|Health Care REITs
60105010|Health Care REITs
601060|Residential REITs
60106010|Multi-Family Residential REITs
60106020|Single-Family Residential REITs
601070|Retail REITs
60107010|Retail REITs
601080|Specialized REITs
60108010|Other Specialized REITs
60108020|Self-Storage REITs
60108030|Telecom Tower REITs
60108040|Timber REITs
60108050|Data Center REITs
6020|Real Estate Management & Development
602010|Real Estate Management & Development
60201010|Diversified Real Estate Activities
60201020|Real Estate Operating Companies
60201030|Real Estate Development
60201040|Real Estate Services
"""

# code -> name, for every level
NAMES: dict[str, str] = {}
for _line in _RAW.strip().splitlines():
    _c, _n = _line.split("|", 1)
    NAMES[_c] = _n

LEVEL_DIGITS = {1: 2, 2: 4, 3: 6, 4: 8}
LEVEL_LABEL = {1: "Sector", 2: "Industry Group", 3: "Industry", 4: "Sub-Industry"}


def codes_at(level: int) -> list[str]:
    d = LEVEL_DIGITS[level]
    return sorted(c for c in NAMES if len(c) == d)


SUB_INDUSTRIES = codes_at(4)          # 所有合法 L4 路徑 (every valid path ends at an L4 code)


def path(code8: str) -> dict:
    """Return the full L1→L4 path for an 8-digit sub-industry code."""
    out = {}
    for lvl, d in LEVEL_DIGITS.items():
        c = code8[:d]
        out[f"L{lvl}"] = {"code": c, "name": NAMES[c]}
    return out


# ---------- name → code matching (for Wikipedia / user CSV labels) ----------
_ALIASES = {
    # older or alternative spellings seen in public constituent lists
    "internet & direct marketing retail": "25503030",
    "general merchandise stores": "25503030",
    "broadline retail": "25503030",
    "hypermarkets & super centers": "30101040",
    "consumer staples merchandise retail": "30101040",
    "airlines": "20302010",
    "trucking": "20304030",
    "railroads": "20304010",
    "data processing & outsourced services": "20202030",
    "internet services & infrastructure": "45102030",
    "movies & entertainment": "50202010",
    "semiconductor equipment": "45301010",
    "semiconductor materials & equipment": "45301010",
    "industrial machinery": "20106020",
    "industrial machinery & supplies & components": "20106020",
    "construction machinery & heavy trucks": "20106010",
    "construction machinery & heavy transportation equipment": "20106010",
    "soft drinks": "30201030",
    "soft drinks & non-alcoholic beverages": "30201030",
    "agricultural products": "30202010",
    "specialized reits": "60108010",
    "residential reits": "60106010",
    "multi-family residential reits": "60106010",
    "single-family residential reits": "60106020",
    "other specialized reits": "60108010",
    "health care reits": "60105010",
    "hotel & resort reits": "60103010",
    "industrial reits": "60102510",
    "office reits": "60104010",
    "retail reits": "60107010",
    "diversified reits": "60101010",
    "real estate services": "60201040",
    "financial exchanges & data": "40203040",
    "thrifts & mortgage finance": "40201050",
    "other diversified financial services": "40201020",
    "multi-sector holdings": "40201030",
    "transaction & payment processing services": "40201060",
    "electronic manufacturing services": "45203020",
    "passenger ground transportation": "20304040",
    "cargo ground transportation": "20304030",
    "rail transportation": "20304010",
}


def _norm(s: str) -> str:
    s = s.lower().replace("&amp;", "&").replace(" and ", " & ")
    return re.sub(r"\s+", " ", s).strip()


_L4_BY_NAME = {_norm(NAMES[c]): c for c in SUB_INDUSTRIES}


def match_sub_industry(name: str, cutoff: float = 0.86) -> str | None:
    """Map a sub-industry name to its 8-digit code (exact → alias → fuzzy)."""
    n = _norm(name or "")
    if not n:
        return None
    if n in _L4_BY_NAME:
        return _L4_BY_NAME[n]
    if n in _ALIASES:
        return _ALIASES[n]
    hit = difflib.get_close_matches(n, list(_L4_BY_NAME), n=1, cutoff=cutoff)
    return _L4_BY_NAME[hit[0]] if hit else None


if __name__ == "__main__":
    counts = {lvl: len(codes_at(lvl)) for lvl in LEVEL_DIGITS}
    assert counts == {1: 11, 2: 25, 3: 74, 4: 163}, counts
    # 前綴一致：每個 L4 的 L1–L3 都存在
    for c8 in SUB_INDUSTRIES:
        for d in (2, 4, 6):
            assert c8[:d] in NAMES, c8
    p = path("45301020")
    assert p["L1"]["name"] == "Information Technology" and p["L4"]["name"] == "Semiconductors"
    assert match_sub_industry("Semiconductors") == "45301020"
    assert match_sub_industry("Internet & Direct Marketing Retail") == "25503030"        # 別名
    assert match_sub_industry("Semiconductor Material and Equipment") == "45301010"      # and→& + 模糊
    assert match_sub_industry("") is None and match_sub_industry("Totally Unknown Widgets") is None
    print(counts, path(match_sub_industry("Semiconductors"))["L4"])
    print("gics_taxonomy selftest OK ✅")
