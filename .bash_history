sudo apt install git -y
git clone https://github.com/amir-jafari/Cloud-Computing.git
cd Cloud-Computing/Deep-Learning-Kit-Installation/Shell-Script-Installation/Ubuntu-22.04/
mv install-22-04-part0-V1.sh ~
mv install-22-04-part1-V1.sh ~
cd~
cd ~
sudo apt purge nvidia* -y
sudo apt remove nvidia-* -y
sudo apt update && sudo apt upgrade -y
sudo apt install g++ freeglut3-dev build-essential libx11-dev libxmu-dev libxi-dev libglu1-mesa libglu1-mesa-dev
sudo add-apt-repository ppa:graphics-drivers/ppa
sudo apt update
ubuntu-drivers devices
sudo apt install ubuntu-drivers-common
ubuntu-drivers devices
sudo ubuntu-drivers devices

nvidia-driver-535-server
apt search nvidia-driver  
sudo apt install libnvidia-common-515 libnvidia-gl-515 nvidia-driver-515 -y
sudo reboot
nvidia-smi
sudo wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/ /"
sudo apt-get update
sudo apt upgrade
sudo apt install cuda-11-8 -y
echo 'export PATH=/usr/local/cuda-11.8/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
sudo ldconfig
nvidia-smi
sudo reboot
while true; do gpustat; sleep 1 ; done
nvidia-smi
nvcc -V
wget https://storage.googleapis.com/cuda-deb/cudnn-local-repo-ubuntu2204-8.6.0.163_1.0-1_amd64.deb
sudo dpkg -i cudnn-local-repo-ubuntu2204-8.6.0.163_1.0-1_amd64.deb
sudo cp /var/cudnn-local-repo-ubuntu2204-8.6.0.163/cudnn-local-FAED14DD-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get install libcudnn8=8.6.0.163-1+cuda11.8
sudo apt-get install libcudnn8-dev=8.6.0.163-1+cuda11.8
sudo apt-get install libcudnn8-samples=8.6.0.163-1+cuda11.8
cat /proc/driver/nvidia/version
nvcc --version
nvidia-smi
sudo apt install libfreeimage3 libfreeimage-dev -y
sudo cp -r /usr/src/cudnn_samples_v8/ $HOME
cd $HOME/cudnn_samples_v8/mnistCUDNN
sudo make
./mnistCUDNN
cd ~
sudo apt install -y python3-pip
sudo apt install build-essential libssl-dev libffi-dev python3-dev -y
sudo apt-get install tcl-dev tk-dev python-tk python3-tk -y
sudo pip3 install --upgrade pip
sudo apt install python3-testresources -y
sudo -H pip3 install tensorflow
sudo -H pip3 install -U scikit-learn
sudo pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
sudo -H pip3 install matplotlib
sudo -H pip3 install pandas
sudo -H pip3 install seaborn
sudo -H pip3 install leveldb
sudo -H pip3 install opencv-python
sudo -H pip3 install pydotplus
sudo -H pip3 install gpustat
sudo -H pip3 install sacred
sudo -H pip3 install pymongo
sudo -H pip3 install openpyxl
sudo -H pip3 install tqdm
sudo -H pip3 install nltk
sudo -H pip3 install pyspellchecker
pip3 install -U pip setuptools wheel
pip3 install -U 'spacy[cuda-autodetect]'
python3 -m spacy download en_core_web_sm
sudo -H pip3 install textacy
sudo -H pip3 install transformers
sudo -H pip3 install datasets
wget https://storage.googleapis.com/cuda-deb/pycharm-community-2022.2.tar.gz
sudo tar -zxf pycharm-community-2022.2.tar.gz
sudo ln -s /home/ubuntu/pycharm-community-2022.2/bin/pycharm.sh pycharm
sudo apt-get install -y p7zip-full
sudo apt install unzip
sudo apt-get install gedit -y
sudo apt-get install python3-gi-cairo
sudo apt install chromium-browser -y
ls
rm -rf cudnn-local-repo-ubuntu2204-8.6.0.163_1.0-1_amd64.deb 
rm install-22-04-part0-V1.sh 
rm install-22-04-part1-V1.sh 
ls
rm -rf Cloud-Computing/
rm -rf pycharm-community-2022.2.tar.gz 
ls
python3
git clone https://github.com/amir-jafari/Deep-Learning.git
cd Deep-Learning/
cd Tensorflow_Basic/
ls
cd MLP/
ls
cd 1_Simple_f_approx/
ls
python3 example_f_approx.py 
pip install optuna
pip install sentencepiece
pip install accelerate -U
pip install accelerate -U install -q -U bitsandbytes
pip install accelerate -U install -q -U bitsandbytes install -q -U git+https://github.com/huggingface/transformers.git
pip install accelerate -U install -q -U bitsandbytes install -q -U git+https://github.com/huggingface/transformers.git install -q -U git+https://github.com/huggingface/peft.git
pip install accelerate -U install -q -U bitsandbytes install -q -U git+https://github.com/huggingface/transformers.git install -q -U git+https://github.com/huggingface/peft.git install -q -U git+https://github.com/huggingface/accelerate.git
pip install accelerate -U install -q -U bitsandbytes install -q -U git+https://github.com/huggingface/transformers.git install -q -U git+https://github.com/huggingface/peft.git install -q -U git+https://github.com/huggingface/accelerate.git install -q -U datasets scipy ipywidgets matplotlib
pip install -q -U bitsandbytes
pip install -q -U git+https://github.com/huggingface/transformers.git
huggingface-cli login
pip install -q wandb -U
pip install --upgrade urllib3
pip install --upgrade chardet
pip install --upgrade pandas
pip install --upgrade urllib3
pip install --upgrade chardet
