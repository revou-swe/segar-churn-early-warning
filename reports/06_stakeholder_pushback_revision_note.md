# Stakeholder Pushback & Revision Note (Checkpoint 3)

One round of stakeholder pushback, role-played with Gemini before showing the product to real stakeholders. Gemini was given the v1.0 dashboard, the memo draft and a persona brief, and asked to push back as each person would.

Each change below was decided on the **validation** window (Feb–Apr 2026) and then confirmed once on the test window, so the fixes are not tuned to the numbers reported.

---

## Round 1 — pushback received

**1. Regional Sales Manager, Jawa Tengah (persona: 15 years in the field, sceptical of dashboards)**
> "Your list has zero warungs on it. Warungs are half my outlets and they're how we get into every kampung. If we stop visiting them, the competitor takes the street, and then the minimarket on the corner follows."

*Assessment:* valid, and confirmed by the bias review: v1.0 reached **0% of churning warungs**.

**2. Head of Sales (persona: numbers-driven, owns the target)**
> "Most of these 200 are 'Low risk' in your own colour coding. Why am I sending reps to supermarkets that are ordering normally? My team will stop trusting this in a month."

*Assessment:* valid. 138 of the 200 slots on the v1.0 September list were outlets under 15% risk. They ranked high only because they are large. Back-test hit rate was 26% vs 48% for today's method.

**3. Finance Controller**
> "You're telling reps to 'push stock' to outlets that already owe us 3 weeks. That's how bad debt starts."

*Assessment:* valid. The v1.0 action for every outlet was a generic retention call.

**4. Area Sales Manager, Jawa Timur**
> "Most of the list is in my region. My reps can't do 80 extra visits."

*Assessment:* partly valid. The concentration is real (Jawa Timur is out of statistical control). But spreading visits evenly would send them where the risk is not. Kept the ranking; escalated the regional problem instead.

## Changes made

| # | Change | Version | Validation effect | Test effect |
|---|---|---|---|---|
| 1 | **Reserve 40 of 200 slots for the highest-risk warungs** not already on the list | v1.1 | warung churners reached 0% → 34% | revenue captured 70.0% → 67.0%; warung reach 0% → 34% |
| 2 | **Minimum 5% churn risk to enter the revenue lane.** Floor chosen on validation as the highest value giving up ≤ 3 pts of revenue capture (grid 0–10%) | v1.2 | hit rate 35.5% → 41.0% | hit rate 37.2% → **43.8%**; revenue captured 67.0% → **68.6%**; supermarket false alarms 28% (v1.0) → 8% |
| 3 | **Action depends on the top reason** (payment plan with Finance for late payers, service recovery for open complaints, range review for shrinking SKUs, etc.), not a generic call | v1.2 | n/a | 21 of 200 October actions are now "agree a payment plan with Finance" |
| 4 | **Alert banner for Jawa Timur** with a recommendation to escalate the competitive response, separate from the visit list | v1.2 | n/a | n/a |
| — | Spread visits evenly by region (request 4) | **not done** | would cut revenue captured | Reason documented in memo; regional review recommended instead |

### Net result, test window (May–Jul 2026), 200 visits/month

| | Today (recency) | v1.0 | **v1.2 deployed** |
|---|---:|---:|---:|
| Revenue-at-risk captured | 43.9% | 70.0% | **68.6%** |
| Hit rate | 47.7% | 25.8% | **43.8%** |
| All churners reached | 57.6% | 31.4% | **53.2%** |
| Warung churners reached | 52.8% | 0.0% | **34.2%** |

v1.2 gives up 1.4 points of revenue capture versus v1.0 to win back trust (hit rate) and coverage (warungs). It still beats today's method on revenue by 25 points. Warung reach remains below today's method; that is the explicit trade-off of revenue weighting, and it is visible on the dashboard.

## What I did not change, and why

- **Model choice.** None of the pushback was about the algorithm; it was about the *policy* built on it. Swapping in gradient boosting would not have answered any of the four concerns.
- **Revenue definition.** The Head of Sales suggested using gross margin instead of revenue. Agreed in principle; margin by outlet is not in the dataset. Logged as a v2 item.
