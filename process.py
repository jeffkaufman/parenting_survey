#!/usr/bin/env python3
import sys
import re
import numpy as np
import scipy
from collections import defaultdict, Counter

def is_na(s):
    return not s or s in ["N/A", "N/a", "N/a--none here", "never", "100", "110"] or any(
        s.startswith(x) for x in [
            "The age where they can",
            "Different for each of these",
            "Wouldn’t",
            "still in utero",
            "No sidewalk",
            "no sidewak",
            "no public transit",
            "I don’t know",
            "Depends on",
            "legally set at 8 in my state",
            "depends on",
            "10? 12? so few good options here.",
            "ha. if only they'd learned.",
        ])

def clean_age(s):
    if is_na(s):
        return float('nan')
    s = s.replace("/", "-")
    s = s.replace(", but depends", "")
    s = s.replace("⁷", "7")
    s = re.sub(" [(].*[)]$", "", s)
    s = s.replace(
        "9 due to threat of CPS; 8 due to threat of stranger danger. I'd let "
        "a younger child play in neighborhood woods alone.", "8")
    if "-" in s:
        return np.average([int(x) for x in s.split("-")])
    if s.endswith(" months"):
        return int(s.replace(" months", ""))/12
    if s.endswith(" weeks"):
        return int(s.replace(" weeks", ""))/52
    if s.endswith(" years"):
        s = s.replace(" years", "")
    return float(s)

def clean_age_range(s):
    if is_na(s):
        return float('nan'), float('nan')
    s = s.replace(" to ", "-")
    if "-" not in s and s.endswith("+"):
        return clean_age(s[:-1]), float('nan')
    if "-" not in s:
        return clean_age(s), clean_age(s)
    early, late = s.split("-")
    return clean_age(early.strip()), clean_age(late.strip())

def clean_area(s):
    if not s:
        return None
    if s in ["Very Urban (tall buildings, no driveways)"]:
        return 1, "very urban"
    if s in ["Moderately Urban (parking is a pain)"]:
        return 2, "moderately urban"
    if s in ["Slightly Urban (multi-family housing is common)",
             "Small town",
             "Medium town; mostly single-family housing, but schools, shops, "
             "restaurants and other destinations are walkable and bikeable"]:
        return 3, "slightly urban"
    if s in [
            "Suburban (almost all single-family housing, few places to go "
            "without driving)",
            "Suburban, but a deliberate cluster of families so many places "
            "to go by feet or bike"]:
        return 4, "suburban"
    if s in ["Exurban (houses widely spaced, you need a car)"]:
        return 5, "exurban"
    if s in ["Rural (houses very far from other houses)"]:
        return 6, "rural"
    raise Exception("Unknown area %r" % s)

def clean_n_children(s):
    if not s: return None
    if s == "I don't have children":
        return "0"
    return s

def clean_gender(s):
    if not s:
        return None
    return s

fname, = sys.argv[1:]
questions = {
    'home_15min': 'Spend fifteen minutes home alone',
    'home_3hr': 'Spend three hours home alone',
    'home_night': 'Spend the night home alone',
    'street_low': 'Cross a low-traffic street',
    'street_medium':'Cross a medium-traffic street',
    'street_busy':'Cross a busy road',
    'school':"Walk to/from school or a friend's house, assuming they"
       " can cross all the streets",
    'backyard':'Play in an unfenced backyard',
    'frontyard':'Play in an unfenced front yard',
    'sidewalk':'Play on the sidewalk in front of their house',
    'playground':'Play at a playground they can walk home from',
    'transit':'Take public transit',
    'bike':'Bike, scooter, or skate around the neighborhood',
}

# question -> age -> count
typicals = defaultdict(Counter)
earlies = defaultdict(Counter)
lates = defaultdict(Counter)

# question -> ages
typicals_list = defaultdict(list)

# records
#   {
#      age,
#      questions, # question -> typical, early, late
#   }
records = []

with open(fname) as inf:
    cols = None

    for line in inf:
        record = {}

        line = line[:-1]
        row = line.split("\t")
        if not cols:
            cols = row
            continue

        def v(s):
            return row[cols.index(s)]
        def v2(s):
            return row[cols.index(s, cols.index(
                "Anything you'd like to clarify about your answers above?"))]

        record["age"] = clean_age(v("What's your age?"))
        record["oldest"] = clean_age(v(
            "How old is your oldest child, if you have one?"))

        record["area"] = clean_area(v("How would you describe your area?"))
        record["childhood_area"] = \
            clean_area(v(
                "How would you describe the area where you grew up? (If "
                "multiple, where you spent the majority of your time from 5-13)"))

        record["n_children"] = clean_n_children(v(
            "How many children do you have, if any?"))

        record["gender"] = clean_gender(v(
            "What's your gender?"))

        question_vals = {}
        for question_slug, question_value in questions.items():
            typical = clean_age(v(question_value))
            early, late = clean_age_range(v2(question_value))

            question_vals[question_slug] = [typical, early, late]
        record["questions"] = question_vals

        records.append(record)

for record in records:
    for question_slug, (typical, early, late) in record["questions"].items():
        if not np.isnan(typical):
            typicals[question_slug][typical] += 1
        if not np.isnan(early):
            earlies[question_slug][early] += 1
        if not np.isnan(late):
            lates[question_slug][late] += 1

for question_slug in questions:
    typical_vals = [record["questions"][question_slug][0]
                    for record in records]
    typical_zscores = scipy.stats.zscore(typical_vals, nan_policy='omit')
    for zscore, record in zip(typical_zscores, records):
        record["questions"][question_slug].append(zscore)

for record in records:
    zscores = [record["questions"][question_slug][-1]
               for question_slug in questions]
    zscores = [x for x in zscores if not np.isnan(x)]
    record['mean_zscore'] = np.mean(zscores) if zscores else float('nan')

# Which question is most representative?
#
# For each person - question pair we have a zscore, and we have the person's
# mean zscore.  Representativeness for a person+question could be:
#
#    abs(zscore(person, question) - zscore(person, *))
#
# And then we could average this over questions?
question_deltas = [] # delta, question
for question_slug in questions:
    deltas = []
    for record in records:
        mean_zscore = record['mean_zscore']
        if np.isnan(mean_zscore):
            continue

        question_zscore = record["questions"][question_slug][-1]
        if np.isnan(question_zscore):
            continue
        deltas.append(abs(question_zscore - mean_zscore))

    question_deltas.append((
        np.mean(deltas), question_slug))

for delta, question_slug in sorted(question_deltas):
    print(delta, question_slug)


def short_label(question_slug):
    if question_slug.startswith("home"):
        return "home"
    elif question_slug in ["backyard", "frontyard", "sidewalk", "playground"]:
        return "play"
    elif question_slug in ["transit", "bike", "school"] or question_slug.startswith("street"):
        return "movement"

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

fig, ax = plt.subplots(constrained_layout=True)
xs = []
ys = []
for record in records:
    if np.isnan(record['mean_zscore']): continue
    if np.isnan(record['age']): continue
    xs.append(record['age'])
    ys.append(record['mean_zscore'])
ax.scatter(xs, ys)
ax.set_title("Caution by age")
ax.set_xlabel("Age")
ax.set_ylabel("Caution z-score")
fig.savefig("caution-by-age-big.png", dpi=180)
plt.close()

fig, ax = plt.subplots(constrained_layout=True)
xs = []
ys = []
for record in records:
    if np.isnan(record['mean_zscore']): continue
    if np.isnan(record['oldest']): continue
    xs.append(record['oldest'])
    ys.append(record['mean_zscore'])
ax.scatter(xs, ys)
ax.set_title("Caution by age of oldest child (parents only)")
ax.set_xlabel("Age of oldest child")
ax.set_ylabel("Caution z-score")
fig.savefig("caution-by-age-of-oldest-big.png", dpi=180)
plt.close()

def tidy_label(variable, record):
    val = record[variable]
    if variable == "area":
        return {
            "very urban": (0, "URB now"),
            "moderately urban": (0, "URB now"),
            "slightly urban": (1, "~URB now"),
            "suburban": (2, "SBR now"),
            "exurban": (3, "EXB now"),
            "rural": (4, "RUR now"),
        }[val[-1]]
    if variable == "childhood_area":
        return {
            "very urban": (1, "URB then"),
            "moderately urban": (1, "URB then"),
            "slightly urban": (1, "URB then"),
            "suburban": (2, "SBR then"),
            "exurban": (3, "EXB then"),
            "rural": (4, "RUR then"),
        }[val[-1]]
    elif variable == "n_children":
        return val + " kids"
    elif variable == "oldest":
        if val <= 2:
            return 1, "oldest 0-2"
        elif val <= 5:
            return 2, "oldest 3-5"
        elif val <= 8:
            return 3, "oldest 6-8"
        elif val <= 12:
            return 4, "oldest 9-12"
        else:
            return 5, "oldest 13+"
    elif variable == "gender":
        return {
            "Female": (1, "female"),
            "Non-binary": (2, "non-binary"),
            "Male": (3, "male"),
        }[val]
    else:
        return val


fig, ax = plt.subplots(constrained_layout=True, figsize=(8,8))
x = []
labels = []
for variable in [
        "childhood_area", "area", "oldest", "n_children", "gender"]:
    def include(record):
        return record[variable] and not np.isnan(record['mean_zscore'])
    for label in sorted(set(tidy_label(variable, record)
                          for record in records
                          if include(record)),
                      reverse=True):
        vals = [record['mean_zscore']
                for record in records
                if include(record) and tidy_label(variable, record) == label]
        if type(label) == type(()):
            _, label = label

        labels.append("%s (n=%s)" % (label, len(vals)))
        x.append(vals)

    if variable != "gender":
        labels.append("")
        x.append([])
box = ax.boxplot(x, labels=labels, vert=False, showfliers=False)
for _, line_list in box.items():
    for line in line_list:
        if line.get_color() != "black":
            line.set_linewidth(line.get_linewidth() * 2)


for n, points in enumerate(x):
    xs_prejitter = points
    ys_prejitter = [n+1 for _ in points]

    xs = xs_prejitter + np.random.normal(0, 0.05, size=(len(points)))
    ys = ys_prejitter + np.random.normal(0, 0.05, size=(len(points)))
    ax.plot(xs, ys, 'b.', alpha=0.2)

ax.set_title("Factors influencing caution")
ax.set_xlabel("Mean z-score")
fig.savefig("factors-big.png", dpi=180)
plt.close()

"""
for variable_name in [
])
    fig, ax = plt.subplots(constrained_layout=True)
    mean_label_row = []
    for question_slug, question_value in questions.items():
        row = []
        for age, count in sorted(counter[question_slug].items()):
            for i in range(count):
                row.append(age)
        mean_label_row.append((np.average(row), question_slug, row))

    labels = [label for (mean, label, row) in sorted(mean_label_row)]
    x = [row for (mean, label, row) in sorted(mean_label_row)]
    ax.set_xlim(xmin=0,xmax=18)
    ax.boxplot(x, labels=labels, vert=False, showfliers=False)
    ax.set_title("Estimates for a %s child" % child_label)
    fig.savefig(child_label + "-big.png", dpi=180)
    plt.close()
"""

"""
fig, ax = plt.subplots(constrained_layout=True)
xs = defaultdict(list)
ys = defaultdict(list)
for i, (_, percentiles) in enumerate(sorted(person_perceniles)):
    for percentile, question_slug in percentiles:
        label = short_label(question_slug)
        xs[label].append(i)
        ys[label].append(percentile*100)
for label in xs:
    ax.scatter(xs[label], ys[label], label=label)
ax.legend()
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
ax.set_title("Relationship of each respondent to others")
ax.set_xlabel("Respondents, from least to most cautious")
ax.set_ylabel("Per-question percentile")
fig.savefig("individual-consistency-big.png", dpi=180)
plt.close()
"""

# n, question_slug
questions_by_mean_typical_age = [
    question_slug
    for (mean_typical_age, question_slug) in sorted(
            (np.average([record["questions"][question_slug][0]
                         for record in records
                         if not np.isnan(record["questions"][
                                 question_slug][0])]),
             question_slug)
            for question_slug in questions)
]

for question_slug, question_value in questions.items():
    fig, axs = plt.subplots(constrained_layout=True, nrows=2, ncols=1,
                            figsize=(10,10),
                            gridspec_kw={'height_ratios': [1, 2]},
                            sharex=True)
    ax = axs[0]
    all_xs = set()
    all_xs |= typicals[question_slug].keys()
    all_xs |= earlies[question_slug].keys()
    all_xs |= lates[question_slug].keys()

    for label, counter in [
            ("typical", typicals[question_slug]),
            ("mature", earlies[question_slug]),
            ("immature", lates[question_slug])]:
        xs = list(sorted(all_xs))
        s = 0
        t = sum(counter.values())
        ys = []
        for x in xs:
            s += counter[x]
            ys.append(100 * s / t)

        ax.plot(xs, ys, label=label)

    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.set_title(question_value.replace(
        ", assuming they can cross all the streets",
        "\n(assuming they can cross all the streets)"))
    ax.legend()

    ax = axs[1]
    x = []
    labels = []
    for variable in [
            "childhood_area", "area", "oldest", "n_children", "gender"]:
        def include(record):
            return record[variable] and not np.isnan(
                record['questions'][question_slug][0])

        for label in sorted(set(tidy_label(variable, record)
                              for record in records
                              if include(record)),
                          reverse=True):
            vals = [record['questions'][question_slug][0]
                    for record in records
                    if include(record) and tidy_label(variable, record) == label]

            if len(vals) < 3:
                continue

            if type(label) == type(()):
                _, label = label

            labels.append("%s (n=%s)" % (label, len(vals)))
            x.append(vals)

        if variable != "gender":
            labels.append("")
            x.append([])
    box = ax.boxplot(x, labels=labels, vert=False, showfliers=False)
    for _, line_list in box.items():
        for line in line_list:
            if line.get_color() == "black":
                line.set_color((0,0,0,.3))
            else:
                line.set_linewidth(line.get_linewidth() * 2)
    for n, points in enumerate(x):
        xs_prejitter = points
        ys_prejitter = [n+1 for _ in points]

        histogram = Counter()
        for point in points:
            histogram[point] += 1

        xs = []
        ys = []
        for point in points:
            xval = point
            yval = n+1
            jitter_scale = histogram[point] - 1
            xval += np.random.normal(0, jitter_scale/150, size=1)[0]
            yval += max(
                min(np.random.normal(0, jitter_scale/100, size=1)[0],
                    .4),
                -.4)

            xs.append(xval)
            ys.append(yval)

        ax.plot(xs, ys, 'b.', alpha=0.2)

    fig.savefig("cdf-" +
                str(questions_by_mean_typical_age.index(question_slug)).zfill(2) +
                "-" + question_slug + "-big.png", dpi=180)
    plt.close()

fig, axs = plt.subplots(constrained_layout=True, nrows=13, ncols=1,
                        figsize=(8,24),
                        sharey=True,
                        sharex=True)
for n, question_slug in enumerate(questions_by_mean_typical_age):
    ax = axs[n]
    all_xs = set()
    all_xs |= typicals[question_slug].keys()
    all_xs |= earlies[question_slug].keys()
    all_xs |= lates[question_slug].keys()

    lines = []
    labels = []
    for label, counter in [
            ("typical", typicals[question_slug]),
            ("mature", earlies[question_slug]),
            ("immature", lates[question_slug])]:
        xs = list(sorted(all_xs))
        s = 0
        t = sum(counter.values())
        ys = []
        for x in xs:
            s += counter[x]
            ys.append(100 * s / t)

        line, = ax.plot(xs, ys, label=label)
        lines.append(line)
        labels.append(label)

    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.set_title(questions[question_slug], loc="left", x=0.01, y=1.0, pad=-16)
    ax.set_xlim(xmax=18, xmin=0)
#plt.figlegend(lines, labels)
fig.savefig("multi-cdf-big.png", dpi=206)
plt.close()

for child_label, counter in [
        ("typical", typicals),
        ("mature", earlies),
        ("immature", lates)]:
    fig, ax = plt.subplots(constrained_layout=True)
    mean_label_row = []
    for question_slug, question_value in questions.items():
        row = []
        for age, count in sorted(counter[question_slug].items()):
            for i in range(count):
                row.append(age)
        mean_label_row.append((np.average(row), question_slug, row))

    labels = [label for (mean, label, row) in sorted(mean_label_row)]
    x = [row for (mean, label, row) in sorted(mean_label_row)]
    ax.set_xlim(xmin=0,xmax=18)
    ax.boxplot(x, labels=labels, vert=False, showfliers=False)
    ax.set_title("Estimates for a %s child" % child_label)
    fig.savefig(child_label + "-big.png", dpi=180)
    plt.close()
