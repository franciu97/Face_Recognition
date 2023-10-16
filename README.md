Installation
Follow these steps to set up the development environment:

1.	Clone the repository: Clone the project repository to your local machine using the following command:

git clone https://github.com/franciu97/Face_Recognition.git

2.	Create a virtual environment: We recommend creating a virtual environment to avoid dependency conflicts. You can do this by running the following command in the project directory:

python3 -m venv virtual_environment_name

3.	Activate the virtual environment: Activate the environment using the appropriate command for your operating system:

On Windows: virtual_environment_name\Scripts\activate
On MacOS and Linux: source virtual_environment_name/bin/activate

4.	Install dependencies: With the virtual environment active, install the required libraries using:

pip install -r requirements.txt

This command installs all necessary libraries, including TensorFlow, OpenCVface_recognition, and others.

5.	Prepare the dataset: Place your reference image in the folder “/path/to/project/Subject” and the images of unknown faces in “/path/to/project/Unknown”.

6.	Train the model: Execute the training script to train the model on your dataset. This process might take some time, depending on your system's capabilities
python3 train_model.py
The script saves the trained model in the project directory.

7.	Run the facial recognition program: Finally, execute the main script that uses your webcam for real-time facial recognition.
python3 main.py

Usage
After executing main.py, position yourself in front of the webcam. The system will attempt to recognize your face based on the reference image. If it matches, it will display your name; otherwise, it will label it as unknown.

Contributing
Contributions make the open-source community such an incredible place to learn, inspire, and create. Any contributions to this project are greatly appreciated.

1.	Fork the Project
2.	Create your Feature Branch (git checkout -b feature/AmazingFeature)
3.	Commit your Changes (git commit -m 'Add some AmazingFeature')
4.	Push to the Branch (git push origin feature/AmazingFeature)
5.	Open a Pull Request

License
Distributed under the MIT License. See LICENSE for more information.
(https://github.com/franciu97/Face_Recognition/assets/39159029/256617d9-9c19-4ca4-9e5a-92c4845064a7)
