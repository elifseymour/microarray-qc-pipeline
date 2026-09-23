**Robotic Microarray Spotting QC Process**

Goal of this project is to establish a biosensor chip QC process

-to eliminate the low-quality chips from downstream processes

-monitor spotting process and see how things go across different spotting runs, and how temperature/humidity/different polymer batches affect spotting quality (spot height and variability) and take measures:

- If CV (for chip-to-chip variability) increases over time – then maybe something is getting worse with the spotter or environmental conditions or polymer coating

- If one antibody’s spot heights have huge variability over different spotting runs – antibody might be going bad or might be unstable or very sensitive to certain environmental conditions

- If failure rate is increasing over time, one or more antibodies might be going bad

If you can narrow down the specific reasons based on correlations etc. then you can pinpoint the problem and avoid unnecessary antibody stock discarding or, waste of spotted chips and resolve/address issue quickly and maintain smooth flow of the lab activities. Overall, this automated analysis makes it easy to monitor the process and ensure high-quality of spotting is maintained. It automatically analyzes different batches, plots data over different spotting runs, helping the scientist with monitoring the process aftereach spotting run, generate QC reports and troubleshoot the process if any anomalies in the chip data shows up and helping gaining insight into the potential causes of the issue.

**Part 1: Deterministic chip QC – Elimination of bad quality chips based on two criteria**

- I need a dataset for 5 different spotting runs (over 5 weeks) and 20 chips per spotting run. Each chip will have 4 different antibodies and 6 replicate spots for each antibody type. For each antibody type, evaluate within-chip spot density variability and antibody surface density and make a chip selection/elimination based on surface density and variability criteria. Prepare a QC report showing each chip’s pass/fail status and reason for failure (spot density or variability). Surface density threshold is 3 ng/mm^2.

- Also keep a record of failure rate for each spotting run (also record it in the QC report) to monitor later in Part 2 over time. (Out of 20 chips what is the failure rate, 10%?, 20%?) If it gets worse it would mean something is progressively getting worse like nozzle of the spotter, one or more specific antibodies going bad over time (we can analyze this by drawing one Ab’s spot density changes vs failure rate in the second part), polymer getting old. In this case there will be an action plan in the second part.

- Again for to be used later in the second part, record chip-to-chip variation for each Ab (as a statistic, like CV) within the batch and track over time for different spotting runs (To see if chip-to-chip variability changes over time, gets worse over time (possibly due to polymer surface getting worse, nozzle getting clogged causing variable droplets, or environmental conditions) or being bad at certain spotting runs sporadically (because of spotter related issues or environmental issues, or one batch of polymer coating being bad from some reason due to personal mistakes for example – so in this case look at correlation between the polymer batch vs CV)

- Other things to record in QC report for the Part 2’s analysis, spotting temperature (both room and within the spotter), spotting humidity (both room and within the spotter), user for spotter, chip ID with specific wafer and polymer coating batch). This data comes from a spotting run ELN record. Usually this info is saved during spotting in a table in ELN system. We used Benchling. So a Benchling notebook (whatever file type that is) would be fed into the dashboard along with the chip Ab density data, a csv file. The csv will have antibody names in the header and surface spot densities in the rows in ng/mm^2.

**Part 2: Batch-to-batch variability of the spotted chips and monitoring spotting process**

- Data from QC reports will be the input data for this part. Spotting batches will be analyzed.

- For example, if one antibody’s spotting height is continuously going down over time and this situation is specific to this Ab, this part of the program should flag this antibody as going bad.

- Trends will be plotted: Effect of temp on average spot height and STD (or CV) over different spotting runs (in this case within a spotting run one antibody type can be pooled over different chips to give a single mean and STD), effect of humidity over different spotting runs, effect of wafer lot, polymer coating batch etc. If things are constant, that is good. If Ab spot height (which corresponds to density) is decreasing what is the cause? Analyze temperature, humidity, polymer batch’s effect on spot height variability.

- Also evaluate chip-to-chip variability metric over different spotting runs. Obtain a CV for chip-to-chip variability for each spotting run (per each Ab) and plot this across different spotting runs.
