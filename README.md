# AI Content Detection

Contains the local training & data generation infrastructure used to create the 5th place solution of the [LLM - Detect AI Generated Text](https://www.kaggle.com/competitions/llm-detect-ai-generated-text/overview) Kaggle competition. See the following for additional information:
* [Writeup describing how the solution works](https://www.kaggle.com/competitions/llm-detect-ai-generated-text/discussion/470093)
* [Best inference/domain adaptaiton code](https://www.kaggle.com/jsday96/multi-context-students/)
* [Generated data](https://www.kaggle.com/datasets/jsday96/ai-content-detection)

This repo is somewhat cluttered with code from early experiments which isn't necessary for reproducing the 5th place solution. The code which directly contributed to it is outlined below.
* **Final classifier training:**
    * TrainTransformer.py
    * TrainTransformer_Short.py
    * TrainMamba.py
    * MultiDataset.py
* **Training models only used for data generation:**
    * Imitating student writing styles:
        * FinetuneStudentImitator.py
        * MergeAdapter.py
    * Models that were the victim of adversarial attack:
        * TrainVictimModels.py
        * Train1DConvResnet.py
        * Train1DConvResnet_DiverseData.py
        * AWP.py
* **Data generation:**
    * GeneratePileCompletions.py
    * GenerateSlimPajamaCompletions.py
    * GenerateTrainingEssays_AdversarialPersuade.py
    * GenerateTrainingEssays_Persuade.py
    * GenerateTrainingEssays_StudentImitatorPersuade.py
* **Data filtering:**
    * SelectTrickyCrawlText.py
    * FilterPileCompletions.py