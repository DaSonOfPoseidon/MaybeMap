# US state name/abbreviation normalization, shared by the spreadsheet loaders and
# the geocoders. The xlsx spells states out ("Missouri") and the CSV export uses
# codes ("MO"); everything downstream wants one or the other, so both formats get
# normalized here rather than at each call site.

US_STATE_NAMES = {
    "AL": "Alabama",
    "AK": "Alaska",
    "AZ": "Arizona",
    "AR": "Arkansas",
    "CA": "California",
    "CO": "Colorado",
    "CT": "Connecticut",
    "DE": "Delaware",
    "DC": "District of Columbia",
    "FL": "Florida",
    "GA": "Georgia",
    "HI": "Hawaii",
    "ID": "Idaho",
    "IL": "Illinois",
    "IN": "Indiana",
    "IA": "Iowa",
    "KS": "Kansas",
    "KY": "Kentucky",
    "LA": "Louisiana",
    "ME": "Maine",
    "MD": "Maryland",
    "MA": "Massachusetts",
    "MI": "Michigan",
    "MN": "Minnesota",
    "MS": "Mississippi",
    "MO": "Missouri",
    "MT": "Montana",
    "NE": "Nebraska",
    "NV": "Nevada",
    "NH": "New Hampshire",
    "NJ": "New Jersey",
    "NM": "New Mexico",
    "NY": "New York",
    "NC": "North Carolina",
    "ND": "North Dakota",
    "OH": "Ohio",
    "OK": "Oklahoma",
    "OR": "Oregon",
    "PA": "Pennsylvania",
    "RI": "Rhode Island",
    "SC": "South Carolina",
    "SD": "South Dakota",
    "TN": "Tennessee",
    "TX": "Texas",
    "UT": "Utah",
    "VT": "Vermont",
    "VA": "Virginia",
    "WA": "Washington",
    "WV": "West Virginia",
    "WI": "Wisconsin",
    "WY": "Wyoming",
}

_ABBREV_BY_NAME = {name.upper(): abbr for abbr, name in US_STATE_NAMES.items()}


def state_abbrev(state: str) -> str:
    # "Missouri" -> "MO". Anything unrecognized is passed through stripped and
    # uppercased, so an odd value still lands somewhere predictable.
    s = (state or "").strip()
    if s.upper() in US_STATE_NAMES:
        return s.upper()
    return _ABBREV_BY_NAME.get(s.upper(), s.upper())


def state_full(state: str) -> str:
    # "MO" -> "Missouri". The geocode caches are keyed on this form, so a lookup
    # hits the same entry whichever spelling the spreadsheet used.
    s = (state or "").strip()
    return US_STATE_NAMES.get(s.upper(), s)
