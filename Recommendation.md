# Recommendation — washing-machine insight from smart-meter data

*For a distribution utility. Based on 20 UK homes (REFIT, 2013–2015); details in RESULTS.md.*

## What the model can and cannot be used for

The model estimates, minute by minute, how much power a home's washing machine draws, using only the
home's total 1-minute consumption. Tested on six homes it had never seen:

| it does well | it does poorly |
|---|---|
| finds **3 out of 4 washing cycles** in the main test home | invents false cycles when other ~2 kW appliances run (kettles, dishwashers, showers) |
| estimates total washing energy over a long period within **~30 %** for a typical home | under-estimates the energy of each individual wash (the long low-power part); month by month the error is larger (median 19–51 % per home) |
| times the washes it finds to within ~2 minutes (median, 5 of 6 test homes) | finds between 18 % and 84 % of washes depending on the home; one home in six was worse than not predicting at all |
| **ranks homes by how often they run hot washes** (rank correlation 0.94 across six unseen homes) | overstates the share of washing in the 16:00–19:00 peak, by 2–18 percentage points per home |

**Suitable uses.** Portfolio-level insight: typical wash timing and frequency across many homes,
ranking households by how often they run hot washes for targeted advice, and the direction of
month-to-month change for groups of homes. In the six test homes, monthly changes went the right
way 12 times in 13, but the model damped them (median 16 % against 20 % true), and its level ran
about 20 % low. So a before/after comparison would likely understate an effect. Peak-hour shares only after correcting them against a
plug-metered panel, because the model puts too much washing energy into the evening peak.

**Not suitable for.** Individual bills or any binding per-home claim; per-wash energy figures;
15- or 30-minute meter data *as input* (a model fed only such data had no per-interval skill;
1-minute input is required, though the results can then be reported per 15 or 30 minutes); homes
with rooftop solar; other countries' appliance stock (e.g. India) without local validation.

## One energy-saving opportunity: lower wash temperatures

**Evidence.** Three quarters of washing-machine electricity (76 %) is spent heating water; 86 % of
washes heat water; a heated wash used a median 0.57 kWh against 0.19 kWh for an unheated one. Only
13 % of washing energy falls in the 16:00–19:00 peak, so shifting washes in time is a much smaller
lever than temperature.

**Action.** Offer households with frequent heated washes a "wash at 30 °C / eco programme"
nudge — in-app message, bill insert or on-demand advice — selected using the model's cycle
detection. I tested this use directly. The model under-counts hot washes, but it ranks homes by
hot-wash frequency well: Spearman 0.94 across six unseen homes, and 0.88 across 13 held-out
periods (few homes, so indicative). So the campaign should take the top of the predicted ranking rather than apply a fixed
count threshold. One caution: the model can assign heating-level power to unheated washes. A home
with no heated wash in its test period got 12 false "hot" washes in 59 weeks, so targeting should be checked
against the calibration panel below.

**Expected size (stated assumptions).** Heating energy scales with the temperature rise; with
~15 °C inlet water, moving a 40 °C wash to 30 °C cuts its heating energy by ~40 %. With 76 % of energy
in heating (mean wash 0.55 kWh), that is ~0.17 kWh saved per heated wash; at ~4 washes a week, 86 % of
them heated, **~30 kWh per household per year**, about £8 at 25 p/kWh — small per home, meaningful
across a customer base, and with a comfort cost close to zero. Feedback programmes typically realise only part of a technical potential; reviews put
appliance-level feedback savings at a few percent of consumption.

**How to measure the impact.** A randomised (or matched) trial: households receiving the nudge vs a
control group, compared before and after (difference-in-differences).
1. **Primary, billing-grade:** change in whole-home consumption from the meter itself.
2. **Attribution:** share of detected washes that contain a heating phase, and estimated heating
   energy per wash, from the model — compared between groups, not trusted in absolute terms.
3. **Calibration panel:** plug monitors on the washing machine in a small subset of homes (10–20) to
   confirm the model's estimated change is real and to correct its bias.

## Additional data that would raise confidence

1. **Reactive power or current harmonics** at 1-second or faster rates: separate the washing
   machine's motor from purely resistive loads (kettles, showers) — the main source of false cycles.
2. **An appliance survey** per home (washer-dryer, dishwasher, electric shower) to explain false
   detections and select suitable homes.
3. **A small plug-metered panel in each new market** (e.g. 10–20 homes in India) to re-train and
   validate locally; Indian machines often lack an internal heater, voltage varies widely, and
   inverters and outages change the signal.
4. **More homes and more recent appliances**: REFIT has 20 homes from one UK town, measured in
   2013–2015.
