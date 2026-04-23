# baseball_lexicon.py

# A sentiment lexicon for analyzing online baseball fan discourse

# Scores are on VADER's scale: -4 (maximally negative) to +4 (maximally positive)
# Use half-integer steps for nuance.
# Phrases with spaces are matched as bigrams/trigrams before tokenization.

# Organising principle:
#   - Performance terms: how fans describe on-field quality
#   - Injury / health terms: implied negative regardless of neutral language
#   - Roster / front office terms: trades, callups, DFA, etc.
#   - Bullpen / pitching: a uniquely baseball source of despair
#   - Fan mood / meta terms: how fans talk about their own feelings
#   - Opponent terms: context-dependent, handled carefully

baseball_lexicon = {
    # 
    # PERFORMANCE — positive
    # 
    "clutch":            3.0,
    "filthy":            2.5,   # a pitch being "filthy" is good
    "nasty":             2.0,   # same — "nasty stuff"
    "electric":          2.5,
    "ace":               2.5,
    "shutdown":          2.0,   # "shutdown inning"
    "no-hitter":         3.5,
    "perfect game":      4.0,
    "cycle":             3.0,
    "walk-off":          3.0,
    "walk off":          3.0,
    "grand slam":        3.0,
    "bomb":              2.0,   # "hit a bomb" = home run
    "dinger":            2.0,
    "gone":              1.5,   # "and it's gone!" home run call
    "launch":            1.5,
    "wheels":            1.5,   # fast runner
    "cannon":            1.5,   # strong arm
    "elite":             2.5,
    "top of the rotation": 2.5,
    "ace stuff":         3.0,
    "swing and a miss":  1.5,   # good for pitchers
    "k":                 0.5,   # context dependent — light positive for strikeout
    "strikeout":         0.5,
    "shutout":           3.0,
    "gem":               3.0,   # "threw a gem"
    "dominant":          2.5,
    "dealing":           2.0,   # "he's dealing" = pitching well
    "locked in":         2.0,
    "on a roll":         2.5,
    "hot streak":        2.5,
    "raking":            2.5,   # hitting well
    "mashing":           2.5,
    "on fire":           2.5,
    "can't miss":        2.0,
    "stud":              2.5,
    "workhorse":         2.0,
    "veteran presence":  1.5,
    "postseason":        2.0,
    "playoffs":          2.0,
    "wild card":         1.5,
    "division lead":     2.5,
    "first place":       2.5,
    "winning streak":    3.0,
    "sweep":             2.5,   # context dependent; assume self-team sweep
    "swept them":        2.5,
    "take the series":   2.0,
    "comeback":          2.5,
    "rally":             2.0,
    "clutch hit":        3.0,
    "big hit":           2.0,
    "huge":              1.5,
    "ws bound":          3.5,
    "world series bound": 3.5,
    "ring":              2.5,   # championship ring

    # 
    # PERFORMANCE — negative
    # 
    "choke":            -3.0,
    "choker":           -3.0,
    "bust":             -2.5,
    "overpaid":         -2.0,
    "washed":           -3.0,   # "he's washed" = past his prime
    "cooked":           -3.0,   # "we're cooked"
    "stick a fork":     -3.0,
    "done":             -1.5,
    "error":            -1.5,
    "errors":           -1.5,
    "costly error":     -2.5,
    "unforced error":   -2.0,
    "mental error":     -2.5,
    "misplay":          -2.0,
    "booted":           -2.0,   # "booted the ball"
    "dropped":          -1.5,
    "meltdown":         -3.0,
    "collapse":         -3.0,
    "folded":           -2.5,
    "blew the lead":    -3.0,
    "blew it":          -3.0,
    "blew the save":    -3.0,
    "lost the lead":    -2.5,
    "gave it away":     -2.5,
    "gave up":          -2.0,
    "winless":          -2.0,
    "losing streak":    -3.0,
    "slump":            -2.0,
    "cold":             -1.5,   # in a slump
    "struggling":       -2.0,
    "can't hit":        -2.0,
    "can't field":      -2.0,
    "can't throw":      -2.0,
    "wild pitch":       -1.5,
    "passed ball":      -1.5,
    "hit by pitch":     -1.0,
    "plunked":          -1.0,
    "ejected":          -2.0,
    "suspended":        -2.0,
    "swept":            -2.5,   # being swept = bad
    "got swept":        -3.0,
    "blown save":       -3.0,
    "loss":             -1.5,
    "losses":           -1.5,
    "last place":       -3.0,
    "basement":         -2.5,
    "cellar":           -2.5,
    "eliminated":       -3.5,
    "out of playoff":   -3.0,
    "fire the manager": -2.5,
    "fire":             -1.0,   # light negative — "fire X" is common
    "fire the gm":      -2.5,
    "terrible":         -2.5,
    "awful":            -2.5,
    "pathetic":         -3.0,
    "embarrassing":     -3.0,
    "embarrassment":    -3.0,
    "disgusting":       -2.5,
    "unwatchable":      -3.0,
    "clown":            -2.0,
    "clown show":       -3.0,
    "dumpster fire":    -3.5,
    "disaster":         -3.0,
    "catastrophe":      -3.0,
    "tank":             -2.0,   # tanking the season
    "tanking":          -2.5,
    "rebuild":          -1.5,   # mixed but often negative in fan context
    "rebuilding":       -1.5,
    "white flag":       -2.5,
    "sell":             -1.5,   # "sell at the deadline"
    "sell off":         -2.0,

    #
    # BULLPEN — almost always negative in fan discourse
    #
    "bullpen":          -0.5,   # slightly negative by default
    "bullpen meltdown": -3.5,
    "bullpen collapse": -3.5,
    "pen":              -0.5,
    "closer":            0.5,   # neutral-positive
    "blown":            -2.5,
    "blew":             -2.5,
    "give it back":     -2.5,
    "relief":           -0.5,
    "bp":               -0.5,
    "pen is cooked":    -3.5,
    "no bullpen":       -3.5,
    "bullpen game":     -2.0,
    "taxed bullpen":    -2.5,
    "overworked":       -2.0,
    "used up":          -2.0,
    "bridge guy":        0.5,
    "set up man":        0.5,

    #
    # INJURIES & HEALTH
    #
    "il":               -2.0,   # "placed on the IL"
    "injured list":     -2.5,
    "day-to-day":       -1.5,
    "dtd":              -1.5,
    "strain":           -1.5,
    "sprain":           -1.5,
    "fracture":         -3.0,
    "torn":             -3.5,
    "surgery":          -3.0,
    "shut down":        -2.5,   # "has been shut down" (pitcher)
    "out for the year": -4.0,
    "season ending":    -4.0,
    "season-ending":    -4.0,
    "missed time":      -2.0,
    "reaggravated":     -3.0,
    "injury-prone":     -2.5,
    "injury":           -2.0,
    "injured":          -2.5,
    "hurt":             -1.5,
    "healthy":           2.0,
    "back healthy":      2.5,
    "activation":        1.5,   # "activated from the IL" — positive
    "activated":         1.5,
    "cleared":           1.5,

    #
    # ROSTER / FRONT OFFICE
    #
    "dfa":              -2.0,   # designated for assignment
    "released":         -1.5,
    "cut":              -1.5,
    "non-tendered":     -1.5,
    "traded away":      -2.0,
    "traded":           -0.5,   # neutral — direction matters
    "acquired":          1.0,
    "signed":            1.0,
    "extension":         2.0,
    "locked up":         2.0,
    "retained":          1.5,
    "promoted":          1.5,
    "callup":            1.5,
    "called up":         1.5,
    "debut":             2.0,
    "prospect":          1.5,
    "top prospect":      2.5,
    "top-100":           2.5,
    "farm system":       0.5,
    "deadline":         -0.5,   # trade deadline — often anxiety
    "trade deadline":   -0.5,
    "sold at deadline": -2.0,
    "buyer":             1.5,
    "seller":           -1.5,
    "ownership":        -1.0,   # typically negative in fan discourse
    "cheap ownership":  -3.0,
    "payroll":          -0.5,
    "cheap":            -2.0,

    #
    # FAN MOOD / META
    #
    "so back":           3.0,   # "we're so back"
    "we're back":        2.5,
    "believe":           2.0,
    "faith":             1.5,
    "bandwagon":        -1.5,
    "bandwagoner":      -1.5,
    "fair weather":     -1.5,
    "heartbreak":       -3.0,
    "heartbreaking":    -3.0,
    "crushing":         -2.5,
    "gutted":           -2.5,
    "devastated":       -3.0,
    "disgusted":        -2.5,
    "embarrassed":      -2.5,
    "hopeless":         -3.0,
    "no hope":          -3.0,
    "nothing to play for": -3.0,
    "couldn't care less": -2.0,
    "gave up on":       -2.0,
    "stopped watching": -2.5,
    "can't watch":      -2.5,
    "hard to watch":    -2.5,
    "painful":          -2.5,
    "suffer":           -2.5,
    "suffering":        -2.5,
    "miserable":        -2.5,
    "excited":           2.5,
    "pumped":            2.5,
    "hyped":             2.5,
    "optimistic":        2.0,
    "cautiously optimistic": 1.5,
    "worried":          -1.5,
    "nervous":          -1.0,
    "concerned":        -1.5,
    "panicking":        -2.0,
    "panic":            -2.0,

    # 
    # SLANG & MEME PHRASES
    # 
    "he's him":          3.0,
    "him":               2.0,
    "not him":          -2.5,
    "fraud":            -3.0,
    "exposed":          -2.5,
    "mid":              -2.0,
    "trash":            -3.0,
    "garbage":          -3.0,
    "ass":              -3.0,
    "dogshit":          -3.5,
    "washed up":        -3.0,
    "carry":             2.5,
    "carried":           2.0,
    "hard carry":        3.0,
    "it's over":        -3.0,
    "joever":           -3.0,
    "gg":                1.0,   # or negative if sarcastic (hard case)
    "lmao":              0.5,   # weak positive / sarcasm indicator
    "lol":               0.5,
    "pain":             -3.0,
    "this team":        -1.5,   # often negative contextually

}
