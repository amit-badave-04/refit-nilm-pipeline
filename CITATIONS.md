# CITATIONS.md — research used, and what I took from each

I read these to choose preprocessing, models and evaluation. Every bibliographic detail was
checked against the publisher, arXiv or the authors' code. Numbers quoted from papers use 8-second
REFIT data unless stated, so they are context, not a leaderboard for my 1-minute results.

## Dataset

1. **Murray, D., Stankovic, L., Stankovic, V. (2017).** An electrical load measurements dataset of
   United Kingdom households from a two-year longitudinal study. *Scientific Data* 4:160122.
   doi:10.1038/sdata.2016.122. Data: doi:10.15129/31da3ece-f902-4e95-a093-e0a9536983c4 (raw),
   doi:10.15129/9ab14b0e-19ac-4279-938f-27f643078cec (cleaned).
   *Used for:* sensors, sampling, units, known gaps, PV houses, IAM coverage, 4 kW fault rule.
2. **Murray, D., Liao, J., Stankovic, L., et al. (2015).** A data management platform for
   personalised real-time energy feedback. *EEDAL 2015* (citation requested by the dataset README).

## Models and protocol

3. **Zhang, C., Zhong, M., Wang, Z., Goddard, N., Sutton, C. (2018).** Sequence-to-point learning
   with neural networks for non-intrusive load monitoring. *AAAI-32*. doi:10.1609/aaai.v32i1.11873.
   *Adopted:* the Seq2Point architecture (five Conv1D layers 30×10, 30×8, 40×6, 50×5, 50×5, Dense
   1024, midpoint output), Adam 1e-3, batch 1,000; the SAE metric; the 20 W WM on-threshold.
4. **D'Incecco, M., Squartini, S., Zhong, M. (2020).** Transfer learning for non-intrusive load
   monitoring. *IEEE Trans. Smart Grid* 11(2):1419–1429. doi:10.1109/TSG.2019.2938068. Code:
   github.com/MingjunZhong/transferNILM.
   *Adopted:* the reference REFIT washing-machine split (train 2, 5, 7, 9, 15, 16, 17; validate 18;
   test 8) and its channel map, used to cross-check my README parser; the 599-sample window, whose
   1-minute equivalent (81) is one of my two candidates; the transfer recipe (freeze the
   convolutional trunk, retrain the dense layers) for the UK→India discussion.
5. **Petralia, A., Charpentier, P., Kadhi, Y., Palpanas, T. (2025).** NILMFormer: non-intrusive
   load monitoring that accounts for non-stationarity. *KDD 2025*. arXiv:2506.05880.
   *Adopted:* the 1-minute REFIT preprocessing (mean resampling, forward-fill ≤ 3 minutes, clipping,
   dropping windows with remaining gaps). Its 1-minute washer table is cited as outside context.
   NILMFormer itself is the first item under "next steps".
6. **Shin, C., Joo, S., Yim, J., Lee, H., Moon, T., Rhee, W. (2019).** Subtask gated networks for
   non-intrusive load monitoring. *AAAI-33*. arXiv:1811.06692.
   *Adopted:* gating the regressed power by a learned on-probability (my M3).
7. **Faustine, A., Pereira, L., Bousbiat, H., Kulkarni, S. (2020).** UNet-NILM: a deep neural
   network for multi-tasks appliances state detection and power estimation in NILM. *NILM'20
   workshop at ACM BuildSys*. doi:10.1145/3427771.3427859.
   *Adopted:* joint state-classification and power-regression heads (supports the M3 design).
8. **Murray, D., Stankovic, L., Stankovic, V., Lulic, S., Sladojevic, S. (2019).** Transferability
   of neural network approaches for low-rate energy disaggregation. *ICASSP 2019*.
   doi:10.1109/ICASSP.2019.8682486.
   *Adopted:* the seen / unseen house framing on REFIT; two-branch state + power networks.
9. **Jia, Z., Yang, L., Zhang, Z., Liu, H., Kong, F. (2021).** Sequence to point learning based on
   bidirectional dilated residual network for non-intrusive load monitoring. *Int. J. Electrical
   Power & Energy Systems* 129:106837. *Considered:* dilated residual trunk; not built (it changes
   several things at once). Evaluated on REDD / UK-DALE, not REFIT.
10. **Kelly, J., Knottenbelt, W. (2015).** Neural NILM: deep neural networks applied to energy
    disaggregation. *ACM BuildSys '15*. doi:10.1145/2821650.2821672.
    *Adopted:* the washing-machine activation rule (on ≥ 20 W, bridge off-gaps < 160 s, discard
    activations < 1,800 s) for cycle detection in the EDA and in the cycle-level metric; the
    seen/unseen evaluation idea. *Not adopted:* the 2,500 W maximum, which flags real 2.55 kW
    heaters in House 4 (replaced by the 13 A plug limit).
11. **Precioso, D., Gómez-Ullate, D. (2023).** Thresholding methods in non-intrusive load
    monitoring. *J. Supercomputing* 79:14039–14062. arXiv:2010.16050.
    *Adopted:* the 1-minute port of Kelly's rule (3 minutes off, 30 minutes on); the "intrinsic
    error" check of the threshold; the warnings that F1 alone misleads and that sparse appliances
    get deceptively low MAE; the combined classification + regression loss.

## Evaluation

12. **Langevin, A., Carbonneau, M.-A., Cheriet, M., Gagnon, G. (2022).** Energy disaggregation
    using variational autoencoders. *Energy and Buildings* 254:111623. arXiv:2103.12177.
    *Adopted:* MAE on active periods (MAE_ON) and daily energy error (EpD). Its REFIT House 8 washer
    results (Seq2Point MAE 21.9 W, F1 61.3 %; VAE 12.0 W, 79.6 %; 8-second data) are cited as context.
13. **Klemenjak, C., Makonin, S., Elmenreich, W. (2020).** Towards comparability in non-intrusive
    load monitoring: on data and performance evaluation. *IEEE ISGT 2020*. arXiv:2001.07708.
    *Adopted:* the noise-to-aggregate ratio; documenting every preprocessing step; reporting how much
    of the data the test covers.
14. **Klemenjak, C., Faustine, A., Makonin, S., Elmenreich, W. (2019).** On metrics to assess the
    transferability of machine learning models in non-intrusive load monitoring. arXiv:1912.06200.
    *Adopted:* seen vs unseen tests, generalisation loss, error/accuracy averaged over unseen homes.
15. **Klemenjak, C., Makonin, S., Elmenreich, W. (2020).** Investigating the performance gap
    between testing on real and denoised aggregates in NILM. arXiv:2008.10985.
    *Adopted:* always test on the real aggregate; report both MAE and NDE.
16. **Batra, N., Kukunuri, R., Pandey, A., et al. (2019).** Towards reproducible state-of-the-art
    energy disaggregation. *ACM BuildSys '19*. doi:10.1145/3360322.3360844 (NILMTK-contrib).
    *Used as:* the reference for benchmark scenarios. I did not depend on NILMTK: it is
    installed from git and nilmtk-contrib pins Python 3.11; the few pieces needed here are short and
    unit-tested.

## Resolution, savings and transfer

17. **Petralia, A., Charpentier, P., Boniol, P., Palpanas, T. (2023).** Appliance detection using
    very low-frequency smart meter time series. *ACM e-Energy '23*. doi:10.1145/3575813.3595198.
    *Used for:* 30-minute evidence (washing-machine detection macro-F1 ≈ 0.61 on REFIT, lower than
    at 1 minute; wet appliances confused with each other).
18. **Zhao, B., Ye, M., Stankovic, L., Stankovic, V. (2020).** Non-intrusive load disaggregation
    solutions for very low-rate smart meter data. *Applied Energy* 268:114949.
    *Used for:* hourly REFIT disaggregation is possible for white goods at the level of daily
    energy, with blurred on/off edges.
19. **Huchtkoetter, J., Reinhardt, A. (2020).** On the impact of temporal data resolution on the
    accuracy of non-intrusive load monitoring. *ACM BuildSys '20*.
    *Used for:* accuracy depends on resolution and generally falls as data get coarser.
20. **Kelly, J., Knottenbelt, W. (2016).** Does disaggregated electricity feedback reduce domestic
    electricity consumption? A systematic review of the literature. arXiv:1605.00962.
    *Used for:* realistic expectations — ~4.5 % savings with appliance-level feedback in volunteer
    studies vs ~3 % with aggregate feedback.
21. **Zimmermann, J.-P., Evans, M., Griggs, J., et al. (2012).** Household Electricity Survey: a
    study of domestic electrical product usage. Intertek report R66141 (DECC/DEFRA/EST).
    *Used for:* the UK benchmark of ~284 washes and ~166 kWh per year (~0.58 kWh per cycle).
22. **Stankovic, L., Stankovic, V., Liao, J., Wilson, C. (2016).** Measuring the energy intensity of
    domestic activities from smart meter data. *Applied Energy* 183:1565–1580.
    *Used for:* laundering as a measurable, shiftable household activity in REFIT homes.
23. **Batra, N., Gulati, M., Singh, A., Srivastava, M. (2013).** It's different: insights into home
    energy consumption in India. *ACM BuildSys '13* (iAWE dataset).
    *Used for:* Indian supply conditions (230 V nominal, 180–260 V measured, long outages).

## Considered and not adopted

* **ELECTRIcity** (Sykiotis et al., *Sensors* 2022): different REFIT split (test house 5) and a
  stated sampling rate inconsistent with REFIT.
* **DiffNILM** (*Sensors* 2023) and LLM-prompting NILM (arXiv 2025): weaker or unreported on REFIT
  washing machines at the time of writing.
* **Data augmentation with synthetic activations** (Kelly & Knottenbelt 2015; Rafiq et al., *IEEE
  TSG* 2021): promising for unseen homes; listed as a next step.
