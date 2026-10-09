# research/ — reading list and protocol notes

The literature behind the methodology: what each paper contributed, and the exact settings taken
from papers and their reference code (`protocol_notes.md`). The full citations, with what was
adopted and what was not, are in [`../CITATIONS.md`](../CITATIONS.md).

The PDFs themselves are not stored here: most are distributed under licences (arXiv's default
licence, publisher agreements) that do not permit redistribution. To read them locally:

```bash
python research/fetch_papers.py      # downloads open-access copies into research/papers/ (git-ignored)
```

## Papers (local file after fetching → citation → what is used from it)

| File (papers/) | Citation | Used for |
|---|---|---|
| 2017_Murray_REFIT_dataset_SciData.pdf | Murray, Stankovic & Stankovic (2017). An electrical load measurements dataset of United Kingdom households from a two-year longitudinal study. *Scientific Data* 4:160122. doi:10.1038/sdata.2016.122 | Dataset facts: 8-s polling, active power (W), CurrentCost clamp + 9 IAMs, ~88% uptime, ~6.4% NaN, IAM coverage 22–55% of load, PV in houses 3/11/21, IAM lag/lead of 2–3 readings |
| 2020_DIncecco_TransferLearningNILM_TSG.pdf | D'Incecco, Squartini & Zhong (2020). Transfer learning for non-intrusive load monitoring. *IEEE Trans. Smart Grid* 11(2):1419–1429. doi:10.1109/TSG.2019.2938068 | Standard REFIT WM split (train 2,5,7,9,15,16,17 / val 18 / test 8); normalisation constants; H8 seq2point WM MAE 16.85 W; cross-dataset transfer (freeze CNN, retrain dense) |
| 2018_Zhang_Seq2Point_AAAI.pdf | Zhang, Zhong, Wang, Goddard & Sutton (2018). Sequence-to-point learning with neural networks for NILM. *AAAI-32*. doi:10.1609/aaai.v32i1.11873 | Seq2Point architecture; SAE definition; 20 W WM on-threshold. Evaluated on UK-DALE/REDD only |
| 2025_Petralia_NILMFormer_KDD.pdf | Petralia, Charpentier, Kadhi & Palpanas (2025). NILMFormer: NILM that accounts for non-stationarity. *KDD 2025*. arXiv:2506.05880 | 1-minute REFIT preprocessing (mean resample, ffill ≤ 3 min, clip 0–10 kW); 1-min washer MAE reference table |
| 2015_Kelly_NeuralNILM_BuildSys.pdf | Kelly & Knottenbelt (2015). Neural NILM: deep neural networks applied to energy disaggregation. *BuildSys '15*. doi:10.1145/2821650.2821672 | WM activation rule: 20 W, min-on 1800 s, min-off 160 s, max 2500 W; seen vs unseen houses; synthetic aggregates |
| 2023_Precioso_Thresholding_JSupercomputing.pdf | Precioso & Gómez-Ullate (2023). Thresholding methods in non-intrusive load monitoring. *J. Supercomputing* 79:14039–14062. arXiv:2010.16050 | Threshold choice via intrinsic error; F1 alone misleads; sparse appliances give deceptively low MAE; multi-task loss |
| 2019_Shin_SubtaskGatedNetworks.pdf | Shin, Joo, Yim, Lee, Moon & Rhee (2019). Subtask gated networks for NILM. *AAAI-33*. arXiv:1811.06692 | Gating the power output by the on-probability |
| (no local copy) | Faustine, Pereira, Bousbiat & Kulkarni (2020). UNet-NILM. *NILM'20 @ BuildSys*. doi:10.1145/3427771.3427859 | Multi-task state + power heads |
| 2021_Jia_BidirectionalDilatedResidual_IJEPES.pdf | Jia, Yang, Zhang et al. (2021). Sequence to point learning based on bidirectional dilated residual network for NILM. *IJEPES* 129:106837 | Dilated residual trunk (optional ablation). Evaluated on REDD/UK-DALE only |
| 2022_Langevin_VAE-NILM_EnergyBuildings.pdf | Langevin, Carbonneau, Cheriet & Gagnon (2022). Energy disaggregation using variational autoencoders. *Energy & Buildings* 254:111623 | MAE_ON and EpD metrics; REFIT H8 WM reference (S2P MAE 21.9 W, F1 61.3%) |
| 2020_Klemenjak_TowardsComparability_ISGT.pdf | Klemenjak, Makonin & Elmenreich (2020). Towards comparability in NILM: on data and performance evaluation. *IEEE ISGT*. arXiv:2001.07708 | Noise-to-aggregate ratio (REFIT H8 ≈ 78%); document preprocessing; report test extent |
| 2019_Klemenjak_TransferabilityMetrics.pdf | Klemenjak, Faustine, Makonin & Elmenreich (2019). On metrics to assess the transferability of ML models in NILM. arXiv:1912.06200 | Seen/unseen definitions; G-loss |
| 2020_Klemenjak_RealVsDenoised.pdf | Klemenjak, Makonin & Elmenreich (2020). Investigating the performance gap between testing on real and denoised aggregates in NILM. arXiv:2008.10985 | Test on the real aggregate; report both MAE and NDE |
| 2019_Murray_Transferability_ICASSP.pdf | Murray, Stankovic, Stankovic, Lulic & Sladojevic (2019). Transferability of neural network approaches for low-rate energy disaggregation. *ICASSP 2019*. doi:10.1109/ICASSP.2019.8682486 | Two-branch state + power networks; unseen REFIT house evaluation (balanced test set, so not comparable) |
| 2023_Petralia_ApplianceDetectionVeryLowFreq_eEnergy.pdf | Petralia, Charpentier, Boniol & Palpanas (2023). Appliance detection using very low-frequency smart meter time series. *e-Energy '23*. doi:10.1145/3575813.3595198 | 30-min evidence: WM detection macro-F1 ≈ 0.61 on REFIT, lower than at 1 min |
| 2020_Zhao_VeryLowRateNILM_AppliedEnergy.pdf | Zhao, Ye, Stankovic & Stankovic (2020). Non-intrusive load disaggregation solutions for very low-rate smart meter data. *Applied Energy* 268:114949 | Hourly REFIT disaggregation; on/off edges blur at low rate |
| 2016_Kelly_DisaggregatedFeedbackReview.pdf | Kelly & Knottenbelt (2016). Does disaggregated electricity feedback reduce domestic electricity consumption? A systematic review. arXiv:1605.00962 | Expected savings from feedback: ~4.5% (volunteer-biased) vs ~3% for aggregate feedback |

## Additional references (no local copy)

- Batra et al. (2019). Towards reproducible state-of-the-art energy disaggregation. *BuildSys '19*. doi:10.1145/3360322.3360844 (NILMTK-contrib).
- NILMBench2026, *BuildSys '26*, github.com/nilmtk/nilmbench.
- Huchtkoetter & Reinhardt (2020). On the impact of temporal data resolution on the accuracy of NILM. *BuildSys '20*.
- Stankovic, Stankovic, Liao & Wilson (2016). Measuring the energy intensity of domestic activities from smart meter data. *Applied Energy* 183:1565–1580.
- Zimmermann et al. (2012). Household Electricity Survey. Intertek report R66141.
- Batra, Gulati, Singh & Srivastava (2013). It's different: insights into home energy consumption in India (iAWE). *BuildSys '13*.

## Dataset

REFIT files, DOIs, licence and checksums: [`../data/README.md`](../data/README.md) and `../data/manifest.json`.
