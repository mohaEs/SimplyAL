# SimplYAL: An Active Learning Interface

This is a React app that can be used to conduct Active Learning locally for image annotation and train models. It can be used by non-professionals, as well. Active Learning is an iterative machine learning approach where the model intelligently selects the most informative samples for human labeling, rather than requiring humans to label every piece of data. The system starts with a small set of labeled images, trains an initial model, and then identifies uncertain or ambiguous images that would most improve the model's performance when labeled. This significantly reduces the time and effort needed for data labeling while maintaining high model quality. Users can simply review and label the suggested images through an intuitive interface, and the model continuously learns and improves from this focused feedback. This approach is particularly valuable when working with large image datasets where labeling everything would be impractical.


## contributors: 
Srikar Kusumanchi (https://github.com/Srikark-17) </br>
Mohammad Eslami (https://github.com/mohaEs)

## Citation: 


## Features

-   Browser-based graphical interface
-   Self-hosted deployment
-   Active learning for image classification
-   Multiple pretrained and custom PyTorch models
-   Multiple uncertainty sampling strategies
-   Automatic iterative retraining
-   Performance visualization
-   Checkpoint management
-   Project import/export

## Architecture

    Browser
        │
    React / Next.js
        │
    FastAPI
        │
    ActiveLearningManager
        ├── PyTorch Models
        ├── Uncertainty Sampling
        ├── Checkpoints
        └── Project Storage


## Installation

### Step 1: System requirements:

verify node, yarn and conda installation:
> node -v
> npm -v
> yarn -v
> conda --version

> which yarn
> which node
> which npm

if above commands return appropriate responsese, go to step 2. otherwise install them:

> sudo apt update 

> sudo apt install libgmp-dev libmpfr-dev libmpc-dev

Install [Node](https://nodejs.org/en/download/), [Yarn](https://classic.yarnpkg.com/en/docs/install/), and [Anaconda](https://www.anaconda.com/products/distribution)


### Step 2: Install SimplyAL:

If above requirements satisfies:

1. Create an environment with conda
   `conda create -n SimplyAL_ENVIRONMENT python=3.11`
2. install the requirements
   `pip install -r backend/requirements.txt`
3. install frontend 
   `yarn --cwd ./frontend`


## Usage

0. open the terminal, activate the enironment
   `conda activate SimplyAL_ENVIRONMENT`
1. run backend
   `python backend/main.py`
2. if not used nohup in step 1, open a new terminal: </br>
   `conda activate SimplyAL_ENVIRONMENT` </br>
   `yarn --cwd ./frontend run dev`</br>
   otherwise </br>
   `yarn --cwd ./frontend run dev`
3. The website should open up on `localhost:3000`. or `http://yourIP:3000`
   use **chrome** to open the link


## User Workflow

An example workflow is shown in this video: 


The user workflow consists of the following steps:

1. **Upload Data**: Upload a CSV file containing the file paths of the images through the web interface. The CSV file can be partially labeled.
2. **Define Labels**: Define the classification labels (e.g., disc-centered and macula-centered). If the CSV file already contains labels for a subset of the images, the software automatically imports the existing annotations.
3. **Select Architecture**: Select a deep learning model architecture.
4. **Configure Settings**: Configure the active learning settings, including the episode size, number of training epochs in each episode, optimizer, learning rate, batch size, and sampling strategy.
5. **Initialize Process**: Initialize the active learning process. If previously annotated samples are available, the system automatically trains the initial (Episode 0) model using the existing labels. Otherwise, the user first annotates an initial seed set before training begins.
6. **Train Model**: Train the model. During training, the software automatically creates a validation subset from the available labeled samples and updates the user interface with learning curves and performance metrics (e.g., accuracy and weighted F1 score). To prevent over-interpretation of unreliable results, the system warns the user when the validation set is too small and reports the validation sample size alongside the displayed metrics.
7. **Rank Images**: Apply the trained model to the remaining unlabeled images. Based on the selected active learning strategy, the software ranks images according to their uncertainty and presents the most informative candidates together with the model's predicted label and confidence score.
8. **Review and Annotate**: Review the suggested images and annotate the selected uncertain samples. If the current model performance is satisfactory, the active learning process can be terminated; otherwise, the user starts the next active learning episode.
9. **Iterate**: Repeat the training and annotation cycle until the desired model performance is achieved.
10. **Export Results**: Export the trained model, accumulated annotations, project configuration, training history, and model predictions for the remaining unlabeled images. All outputs are packaged into a compressed ZIP archive that can be downloaded directly through the web interface. Depending on the dataset size and the generated outputs, the export process may take from a few seconds to several minutes.
11. **Cleanup**: After the project has been successfully exported, users are encouraged to remove temporary project files from the server to release storage resources. This is particularly beneficial when working with large image datasets or when multiple projects are hosted on the same workstation or institutional server.


## Sampling Strategies

- `least_confidence`: Selects samples where the model's highest predicted class probability is lowest, indicating uncertainty
- `margin`: Picks samples with the smallest difference between the top two predicted class probabilities, suggesting confusion between classes
- `entropy`: Chooses samples with highest entropy in their prediction probabilities, indicating maximum uncertainty across all classes
- `diversity`: Selects samples that are most different from the currently labeled data in feature space, ensuring varied training examples


## Clinical Deployment

SimplyAL is designed for self-hosted deployment on secure institutional
infrastructure. Users access the software through a web browser while
training, GPUs, and image storage remain centralized on a workstation or
server.


## License

BSD 3-Clause License.

## Support

Please use GitHub Issues for bug reports and feature requests.

## Disclaimer
**Liability Disclaimer**: This software is provided for research and educational purposes only. The software is supplied "as is," without warranty of any kind, and the authors assume no responsibility for any errors, data loss, or damages arising from its use. Users are responsible for validating results. 

**Clinical Disclaimer**: This software is intended for research and educational use only. It is not a medical device and must not be used as the sole basis for clinical diagnosis or treatment decisions. Users are responsible for validating results. 