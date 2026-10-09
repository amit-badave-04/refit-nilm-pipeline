"""Download open-access copies of the papers listed in research/README.md into research/papers/.

The PDFs are not stored in the repository: most are distributed under licences (arXiv's default
licence, publisher agreements) that do not permit redistribution. Fetching them from their
official open-access locations keeps the reading list reproducible.

Usage: python research/fetch_papers.py
"""
import urllib.request
from pathlib import Path

OUT = Path(__file__).resolve().parent / "papers"

PAPERS = {
    "2015_Kelly_NeuralNILM_BuildSys.pdf": "https://arxiv.org/pdf/1507.06594",
    "2016_Kelly_DisaggregatedFeedbackReview.pdf": "https://arxiv.org/pdf/1605.00962",
    "2017_Murray_REFIT_dataset_SciData.pdf": "https://www.nature.com/articles/sdata2016122.pdf",
    "2018_Zhang_Seq2Point_AAAI.pdf": "https://arxiv.org/pdf/1612.09106",
    "2019_Klemenjak_TransferabilityMetrics.pdf": "https://arxiv.org/pdf/1912.06200",
    "2019_Murray_Transferability_ICASSP.pdf": "https://strathprints.strath.ac.uk/66112/14/Murray_etal_ICASSP2019_Transferability_of_neural_networks_approaches_for_low_rate_energy.pdf",
    "2019_Shin_SubtaskGatedNetworks.pdf": "https://arxiv.org/pdf/1811.06692",
    "2020_DIncecco_TransferLearningNILM_TSG.pdf": "https://arxiv.org/pdf/1902.08835",
    "2020_Klemenjak_RealVsDenoised.pdf": "https://arxiv.org/pdf/2008.10985",
    "2020_Klemenjak_TowardsComparability_ISGT.pdf": "https://arxiv.org/pdf/2001.07708",
    "2020_Zhao_VeryLowRateNILM_AppliedEnergy.pdf": "https://strathprints.strath.ac.uk/72013/1/Zhao_etal_AE_2020_Non_intrusive_load_disaggregation_solutions.pdf",
    "2021_Jia_BidirectionalDilatedResidual_IJEPES.pdf": "https://arxiv.org/pdf/2006.00250",
    "2022_Langevin_VAE-NILM_EnergyBuildings.pdf": "https://arxiv.org/pdf/2103.12177",
    "2023_Petralia_ApplianceDetectionVeryLowFreq_eEnergy.pdf": "https://arxiv.org/pdf/2305.10352",
    "2023_Precioso_Thresholding_JSupercomputing.pdf": "https://arxiv.org/pdf/2010.16050",
    "2025_Petralia_NILMFormer_KDD.pdf": "https://arxiv.org/pdf/2506.05880",
}


def main() -> None:
    OUT.mkdir(exist_ok=True)
    for name, url in PAPERS.items():
        dest = OUT / name
        if dest.exists() and dest.read_bytes()[:5] == b"%PDF-":
            print(f"have  {name}")
            continue
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        try:
            data = urllib.request.urlopen(req, timeout=120).read()
        except OSError as exc:
            print(f"FAIL  {name}: {exc}")
            continue
        if data[:5] != b"%PDF-":
            print(f"FAIL  {name}: response was not a PDF")
            continue
        dest.write_bytes(data)
        print(f"got   {name} ({len(data) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
