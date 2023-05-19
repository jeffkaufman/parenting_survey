#!/usr/bin/env python3
import sys
import numpy as np
from collections import defaultdict, Counter

def is_na(s):
    return not s or s in ["N/A", "N/a", "N/a--none here", "never", "100", "110"] or any(
        s.startswith(x) for x in [
            "The age where they can",
            "Different for each of these",
            "Wouldn’t",
        ])

def clean_age(s):
    if is_na(s):
        return None
    if "-" in s:
        return np.average([int(x) for x in s.split("-")])
    s = s.replace(" (almost 10)", "")
    if s.endswith(" months"):
        return int(s.replace(" months", ""))/12
    if s.endswith(" weeks"):
        return int(s.replace(" weeks", ""))/52
    if s.endswith(" years"):
        s = s.replace(" years", "")
    return float(s)

def clean_age_range(s):
    if is_na(s):
        return None, None
    s = s.replace(" to ", "-")
    if "-" not in s and s.endswith("+"):
        return clean_age(s[:-1]), None
    if "-" not in s:
        return clean_age(s), clean_age(s)
    early, late = s.split("-")
    return clean_age(early.strip()), clean_age(late.strip())

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

with open(fname) as inf:
    cols = None

    for line in inf:
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
        
        age = v("What's your age?")
        if age:
            age = int(age)
        else:
            age = None

        oldest = clean_age(v("How old is your oldest child, if you have one?"))

        for question_slug, question_value in questions.items():
            typical = clean_age(v(question_value))
            early, late = clean_age_range(v2(question_value))

            if typical is not None:
                typicals[question_slug][typical] += 1
            if early is not None:
                earlies[question_slug][early] += 1
            if late is not None:
                lates[question_slug][late] += 1
        
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

for question_slug, question_value in questions.items():
    fig, ax = plt.subplots(constrained_layout=True)
    
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
    fig.savefig(question_slug + "-big.png", dpi=180)
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
        
        
        
