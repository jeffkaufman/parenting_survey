#!/usr/bin/env python3
import sys
import re
import numpy as np
import scipy
from collections import defaultdict, Counter

def is_na(s):
    s = s.strip()
    return not s or s in [
        "N/A", "N/a", "N/a--none here", "100", "110",
        "I do not trust drivers in Somerville.",
        "With a friend 12 Alone, unsure when he'll feel ready.",
        "We don't have a sidewalk very close to our house",
        "Depend on the kid maybe, depending on yard and traffic",
    ] or any(
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
            "I don't understand this question",
            "2I don't understand this question",
            "with supervision, like 4",
        ])

def clean_age(s):
    if is_na(s):
        return float('nan')

    for f, r in [
            # Handle verbose answers
            ("8 for our neighborhood, 6-7 for a more suburban area ",
             "8"),
            ("12 ? Depends on which one! I do feel like it depends on the "
             "particular street as I think there are good and bad crossings. Ie "
             "busy roads by highways: super dangerous; Memorial Drive — is fast "
             "but crossings close to Harvard Sq are pretty safe for peds.",
             "12"),
            (
                "I’m assuming this is unsupervised? I’m having trouble "
                "imaging an unfenced backyard in my neighborhood. Depends "
                "on kkd if you’re worried they’ll wander off! 3? 4? My kid "
                "would never wander off but I know kids who are runners",
                "3"),
            ("I do not trust drivers in Somerville. 8", "8"),
            ("8 except I do not trust drivers in Somerville.", "8"),
            ("8, depends on if other adults are known to be present", "8"),
            ("10 but more dependent on neighborhood than child", "10"),
            ("6. This is also the legal minimum age where I live", "6"),
            ("9 due to threat of CPS; 8 due to threat of stranger danger. "
             "I'd let a younger child play in neighborhood woods alone.", "8"),
            ("7 if w/in quarter mile, 9 if more like a mile", "9"),
            ("8, but I’m not sure my kid would be ready", "8"),
            ("12 depends on kid and environment", "12"),
            ("12 depends on neighborhood", "12"),
            ('7 with a crosswalk signal', "7"),
            # remove qualifiers
            ("(unsupervised, you mean?)", ""),
            ("Wildly child dependent.", ""),
            ("almost ", ""),
            (", but depends", ""),
            # I'm interpreting "never" to mean "not while they're a kid"
            ("I think of McGrath and say never", "18"),
            ("Never — this is not something I believe to be appropriate",
            "18"),
            ("never", "18"),
            ("no", "18"),
            # alterantive ways of writing things
            ("/", "-"),
            ("⁷", "7"),
            # treat 8+ etc as 8
            ("+", ""),
            # remove uncertainty markers
            ("?", ""),
    ]:
        s = s.replace(f, r)
    s = re.sub(" [(].*[)]$", "", s)
    s = s.strip()
    if "-" in s:
        return np.average([int(x) for x in s.split("-")])
    if s.endswith(" months"):
        return int(s.replace(" months", ""))/12
    if s.endswith(" weeks"):
        return int(s.replace(" weeks", ""))/52
    if s.endswith(" years"):
        s = s.replace(" years", "")
    if s.endswith(" years old"):
        s = s.replace(" years old", "")
    return float(s)

def clean_age_range(s):
    s = s.replace("5-10 (for ~0.25 mile), 7-12 (for ~1 mile)", "7-12")
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
             "Small Town which is walkable unless you want to leave town",
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
        
        if record["n_children"] is None:
            record["is_parent"] = float("nan")
        elif record["n_children"] == "0":
            record["is_parent"] = 2
        elif record["n_children"] in ["1", "2", "3", "4", "5+"]:
            record["is_parent"] = 1
        else:
            assert False, record["n_children"]
            
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
    typical_mean = np.mean([x for x in typical_vals if not np.isnan(x)])
    typical_distance_from_mean_years = [
        val - typical_mean
        for val in typical_vals
    ]
    for zscore, distance_years, record in zip(
            typical_zscores, typical_distance_from_mean_years, records):
        record["questions"][question_slug].append(zscore)
        record["questions"][question_slug].append(distance_years)


for record in records:
    zscores = [record["questions"][question_slug][3]
               for question_slug in questions]
    zscores = [x for x in zscores if not np.isnan(x)]
    record['mean_zscore'] = np.mean(zscores) if zscores else float('nan')
    
    distances = [record["questions"][question_slug][4]
                 for question_slug in questions]
    distances = [x for x in distances if not np.isnan(x)]
    record['mean_distance_years'] = \
        np.mean(distances) if distances else float('nan')
    record["highlight"] = None
    if (record["age"] == 37 and
        record["gender"] == "Male" and
        record["area"][1] == "moderately urban" and
        record["childhood_area"][1] == "slightly urban"):
        record["highlight"] = 'b'
    if record["age"] == 7:
        record["highlight"] = 'r'
    if record["age"] == 9 and  record["area"][1] == "moderately urban":
        record["highlight"] = 'r'
              
    
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

print("Responses: %s" % len(records))

print("Gender counts:")
genders = Counter()
for record in records:
    if record["gender"]:
        genders[record["gender"]] += 1
for gender in ["Male", "Female", "Non-binary"]:
    print("  %s %s (%.0f%%)" % (
        gender, genders[gender], 100 * genders[gender] / sum(genders.values())))    

print("Area counts:")
areas = Counter()
for record in records:
    if record["area"]:
        areas[record["area"]] += 1
for area in areas:
    print("  %s %s (%.0f%%)" % (
        area, areas[area], 100 * areas[area] / sum(areas.values())))    

print("Childhood Area counts:")
childhood_areas = Counter()
for record in records:
    if record["childhood_area"]:
        childhood_areas[record["childhood_area"]] += 1
for childhood_area in childhood_areas:
    print("  %s %s (%.0f%%)" % (
        childhood_area, childhood_areas[childhood_area], 100 * childhood_areas[childhood_area] / sum(childhood_areas.values())))    

print("N children counts:")
n_childrens = Counter()
for record in records:
    if record["n_children"]:
        n_childrens[record["n_children"]] += 1
for n_children in n_childrens:
    print("  %s %s (%.0f%%)" % (
        n_children, n_childrens[n_children], 100 * n_childrens[n_children] / sum(n_childrens.values())))    

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

fig, ax = plt.subplots(constrained_layout=True)
xs = []
ys = []

all_ages = []
ages = Counter()
for record in records:
    if np.isnan(record['age']): continue
    ages[record['age']] += 1
    all_ages.append(record['age'])
for age, count in ages.items():
    xs.append(age)
    ys.append(count)
ax.set_title("Age distribution of respondents")
ax.set_xlabel("Age")
ax.set_ylabel("Number of respondents")
ax.set_ylim(ymin=0, ymax=18)
ax.scatter(xs, ys)
fig.savefig("parenting-survey-age-distibution-big.png", dpi=180)
plt.close()

fig, ax = plt.subplots(constrained_layout=True)
xs = []
ys = []

ages = Counter()
for record in records:
    if np.isnan(record['age']): continue
    if np.isnan(record['mean_distance_years']): continue
    xs.append(record['age'])
    ys.append(record['mean_distance_years'])
ax.set_title("Relation between age and higher-age responses")
ax.set_xlabel("Respondent age")
ax.set_ylabel("Mean years later than average")
ax.scatter(xs, ys)
fig.savefig("parenting-survey-age-vs-relative-big.png", dpi=180)
plt.close()

fig, ax = plt.subplots(constrained_layout=True)
xs = []
ys = []

ages = Counter()
for record in records:
    if np.isnan(record['oldest']): continue
    if np.isnan(record['mean_distance_years']): continue
    xs.append(record['oldest'])
    ys.append(record['mean_distance_years'])
ax.set_title("Relation between age of oldest child and higher-age responses")
ax.set_xlabel("Respondent's oldest child")
ax.set_ylabel("Mean years later than average")
ax.scatter(xs, ys)
fig.savefig("parenting-survey-oldest-vs-relative-big.png", dpi=180)
plt.close()

print("Median age: %s" % np.median(all_ages))

fig, ax = plt.subplots(constrained_layout=True)
xs = []
ys = []
all_oldests = []
oldests = Counter()
for record in records:
    if np.isnan(record['oldest']): continue
    oldests[round(record['oldest'])] += 1
    all_oldests.append(record['oldest'])
for oldest, count in oldests.items():
    xs.append(oldest)
    ys.append(count)
ax.set_title("Distribution of the oldest child of respondents")
ax.set_xlabel("Oldest child, if a parent")
ax.set_ylabel("Number of respondents")
ax.set_ylim(ymin=0, ymax=24)
ax.scatter(xs, ys)
fig.savefig("parenting-survey-oldest-distibution-big.png", dpi=180)
plt.close()

print("Fraction oldest under 18: %.0f%%" % (
    100 * sum(1 for x in all_oldests if x < 18) /
    len(all_oldests)))

print("Median oldest: %s" % np.median(all_oldests))

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
fig.savefig("parenting-survey-caution-by-age-big.png", dpi=180)
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
fig.savefig("parenting-survey-caution-by-age-of-oldest-big.png", dpi=180)
plt.close()

oldest_at_birth = []
for record in records:
    if np.isnan(record['age']): continue
    if np.isnan(record['oldest']): continue
    oldest_at_birth.append(record['age'] - record['oldest'])
print("Mean age at first child: %s" % np.mean(oldest_at_birth))
print("Median age at first child: %s" % np.median(oldest_at_birth))

fig, ax = plt.subplots(constrained_layout=True)
sorted_areas = []
ys = []
xs = []
for n, area in sorted(areas):
    ys.append(area)
    sorted_areas.append(area)
    xs.append(areas[n, area])
y_pos = np.arange(len(ys))
ax.barh(y_pos, xs, align='center')
ax.set_yticks(y_pos)
ax.set_yticklabels(ys)
ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel('Respondents')
ax.set_title('Location distribution of respondents')
fig.savefig("parenting-survey-current-location-big.png", dpi=180)
plt.close()

fig, ax = plt.subplots(constrained_layout=True)
ys = []
xs = []
for n, childhood_area in sorted(childhood_areas):
    ys.append(childhood_area)
    xs.append(childhood_areas[n, childhood_area])
y_pos = np.arange(len(ys))
ax.barh(y_pos, xs, align='center')
ax.set_yticks(y_pos)
ax.set_yticklabels(ys)
ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel('Respondents')
ax.set_title('Childhood location distribution of respondents')
fig.savefig("parenting-survey-childhood-location-big.png", dpi=180)
plt.close()

fig, ax = plt.subplots(constrained_layout=True)
area_scatter_counts = Counter()
for record in records:
    if record["area"] and record["childhood_area"]:
        area_scatter_counts[record["area"][0],
                            record["childhood_area"][0]] += 1

ys = []
xs = []
sizes= []
for (y, x), count in area_scatter_counts.items():
    xs.append(x)
    ys.append(y)
    sizes.append(count*30)

plt.xticks([n+1 for n in range(len(sorted_areas))], sorted_areas,
           rotation=45, ha='right')
plt.yticks([n+1 for n in range(len(sorted_areas))], sorted_areas)
ax.set_xlabel('Childhood area')
ax.set_ylabel('Current area')

ax.set_title('Relation between current area and childhood area')

ax.scatter(xs, ys, sizes=sizes)

fig.savefig("parenting-survey-location-relation-big.png", dpi=180)
plt.close()        


fig, ax = plt.subplots(constrained_layout=True)
ys = []
xs = []
for n_children in sorted(n_childrens):
    ys.append(n_children)
    xs.append(n_childrens[n_children])
y_pos = np.arange(len(ys))
ax.barh(y_pos, xs, align='center')
ax.set_yticks(y_pos)
ax.set_yticklabels(ys)
ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel('Respondents')
ax.set_title('Number of children')
fig.savefig("parenting-survey-number-of-children-big.png", dpi=180)
plt.close()



def tidy_label(variable, record):
    val = record[variable]
    if variable == "is_parent":
        #print(val)
        return {
            1: "parents",
            2: "non-parents",
        }[val]
    if variable == "area":
        return {
            "very urban": (0, "urban now"),
            "moderately urban": (0, "urban now"),
            "slightly urban": (0, "urban now"),
            "suburban": (2, "suburban now"),
            "exurban": (3, "exurban or rural now"),
            "rural": (3, "exurban or rural now"),
        }[val[-1]]
    if variable == "childhood_area":
        return {
            "very urban": (1, "urban then"),
            "moderately urban": (1, "urban then"),
            "slightly urban": (1, "urban then"),
            "suburban": (2, "suburban then"),
            "exurban": (3, "exurban then"),
            "rural": (4, "rural then"),
        }[val[-1]]
    elif variable == "n_children":
        if val == "0":
            return ("0", "no kids")
        else:
            if val in ["4", "5+"]:
                val = "4+"
            return (val, val + " kids")
    elif variable == "age":
        if 7 <= val <= 9:
            return 7, "7-9"
        elif 25 <= val <= 29:
            return 25, "25-29" 
        elif 30 <= val <= 34:
            return 30, "30-34"
        elif 35 <= val <= 39:
            return 35, "35-39"  
        elif 40 <= val <= 44:
            return 40, "40-44"
        elif 45 <= val <= 49:
            return 45, "45-49"
        elif val >= 50:
            return 50, "50+"
        return val
    elif variable == "oldest":
        if val <= 3:
            return 1, "oldest 0-3"
        elif val <= 5:
            return 2, "oldest 4-5"
        elif val <= 7:
            return 3, "oldest 6-7"
        elif val <= 9:
            return 4, "oldest 8-9"
        elif val <= 12:
            return 5, "oldest 10-12"
        elif val <= 18:
            return 6, "oldest 13-18"
        else:
            return 7, "oldest 18+"
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
box = ax.boxplot(x, labels=labels, vert=False, showfliers=False, showmeans=True)
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

ax.set_title("Factors predicting higher-age responses")
ax.set_xlabel("Mean z-score: larger values indicate higher-age responses")
fig.savefig("parenting-survey-factors-big.png", dpi=180)
plt.close()


for figlabel, factors, figsize in [
        ("areas", ("childhood_area", "area"), (8,3)),
        ("kids", ("oldest", "n_children", "is_parent"), (8,5)),
        ("gender", ("gender", ), (8,2)),
        ("age", ("age", ), (8,3)),
]:
    fig, ax = plt.subplots(constrained_layout=True, figsize=figsize)
    x = []
    labels = []
    for variable in factors:
        def include(record):
            return record[variable] and (
                type(record[variable]) != type(0.0) or
                not np.isnan(record[variable])
            ) and not np.isnan(record['mean_distance_years'])
        for label in sorted(set(tidy_label(variable, record)
                                for record in records
                                if include(record)),
                            reverse=True):
            vals = [record['mean_distance_years']
                    for record in records
                    if include(record) and tidy_label(variable, record) == label]
            if type(label) == type(()):
                _, label = label

            if label == "no kids": continue

            labels.append("%s (n=%s)" % (label, len(vals)))
            x.append(vals)

        if variable != factors[-1]:
            labels.append("")
            x.append([])
    box = ax.boxplot(x, labels=labels, vert=False, showfliers=False, showmeans=True)
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

    ax.set_title("Factors predicting higher-age responses")
    ax.set_xlabel("Mean years later than average")
    fig.savefig("parenting-survey-factors-%s-age-distance-big.png" %
                figlabel, dpi=180)
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
fig.savefig("parenting-survey-individual-consistency-big.png", dpi=180)
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
            ("immature", lates[question_slug]),
            ("mature", earlies[question_slug]),
    ]:
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
    box = ax.boxplot(x, labels=labels, vert=False, showfliers=False,
                     showmeans=True)
    for _, line_list in box.items():
        for line in line_list:
            line.set_color((0,0,0,.3))
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
    ax.set_xlim(xmax=18, xmin=0)

    fig.savefig("parenting-survey-cdf-" +
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
            ("immature", lates[question_slug]),
            ("mature", earlies[question_slug]),
    ]:
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
fig.savefig("parenting-survey-multi-cdf-big.png", dpi=206)
plt.close()

fig, ax = plt.subplots(constrained_layout=True, nrows=1, ncols=1,
                        figsize=(8,4),
                        sharey=True,
                        sharex=True)
for n, question_slug in enumerate(questions_by_mean_typical_age):
    if question_slug != "transit":
        continue

    all_xs = set()
    all_xs |= typicals[question_slug].keys()
    all_xs |= earlies[question_slug].keys()
    all_xs |= lates[question_slug].keys()

    lines = []
    labels = []
    for label, counter in [
            ("typical", typicals[question_slug]),
            ("immature", lates[question_slug]),
            ("mature", earlies[question_slug]),
    ]:
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
    ax.set_title(
        "Age at which a child can first handle taking public transit solo")
    ax.set_xlim(xmax=18, xmin=0)
    ax.legend()

fig.savefig("parenting-survey-transit-cdf-big.png", dpi=206)
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
            ("immature", lates[question_slug]),
            ("mature", earlies[question_slug]),
    ]:
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

    for record in records:
        if not record["highlight"]: continue

        typical, mature, immature, *_ = record["questions"][question_slug]
        ax.axvline(x=typical, color = record["highlight"])

    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.set_title(questions[question_slug], loc="left", x=0.01, y=1.0, pad=-16)
    ax.set_xlim(xmax=18, xmin=0)
#plt.figlegend(lines, labels)
fig.savefig("parenting-survey-multi-cdf-highlight-big.png", dpi=206)
plt.close()

fig, ax = plt.subplots(constrained_layout=True)
ys = []
xs = []
for n, question_slug in enumerate(questions_by_mean_typical_age):
    q = questions[question_slug]
    q = q.replace(", assuming they can cross all the streets", "")
    ys.append(q)
    xs.append(np.mean([
        typical_age for record in records
        if not np.isnan(
                typical_age := record["questions"][question_slug][0])]))

y_pos = np.arange(len(ys))
ax.barh(y_pos, xs, align='center')
for i in range(len(xs)):
    plt.text(xs[i] - 0.1 , i + 0.15,
             "%.1f"%xs[i],
             horizontalalignment='right', color="w")
ax.set_xlim(xmin=0,xmax=18)
ax.set_yticks(y_pos)
ax.set_yticklabels(ys)
ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel('Mean age')
ax.set_title('Activities by age at which children typically can handle them solo',
             loc="left", x=-1.1)
fig.savefig("parenting-survey-mean-typical-age-big.png", dpi=180)
plt.close()

fig, ax = plt.subplots(constrained_layout=True)
for label, pos, color in [
        ("immature", 2, 'C1'),
        ("typical", 0, 'C0'),
        ("mature", 1, 'C2'),
]: 
    ys = []
    xs = []
    for n, question_slug in enumerate(questions_by_mean_typical_age):
        q = questions[question_slug]
        q = q.replace(", assuming they can cross all the streets", "")
        ys.append(q)
        xs.append(np.mean([
            typical_age for record in records
            if not np.isnan(
                    typical_age := record["questions"][question_slug][pos])]))
    y_pos = np.arange(len(ys))
    ax.barh(y_pos, xs, align='center', color=color, label=label)

    if label != "typical":
        for i in range(len(xs)):
            plt.text(xs[i] - 0.1 , i + 0.15,
                     "%.1f"%xs[i],
                     horizontalalignment='right', color="w")
ax.set_xlim(xmin=0,xmax=18)
ax.legend()
y_pos = np.arange(len(ys))
ax.set_yticks(y_pos)
ax.set_yticklabels(ys)
ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel('Mean age')
ax.set_title('Activities by age at which children typically can handle them solo',
             loc="left", x=-1.1)
fig.savefig("parenting-survey-mean-multi-age-big.png", dpi=180)
plt.close()

for child_label, counter in [
        ("typical", typicals),
        ("immature", lates),
        ("mature", earlies),
]:
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
    fig.savefig("parenting-survey-" + child_label + "-big.png", dpi=180)
    plt.close()

# exporting
import json
records.sort(key=lambda record: record["mean_distance_years"])
for record in records:
    for question_slug in questions:
        record["questions"][question_slug] = {
            "typical": record["questions"][question_slug][0],
            "mature": record["questions"][question_slug][1],
            "immature": record["questions"][question_slug][2],
            "zscore": record["questions"][question_slug][3],
            "years_above_mean": record["questions"][question_slug][4],
        }
with open("export.json", "w") as outf:
    json.dump(records, outf, sort_keys=True, indent=2)
