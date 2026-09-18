# Advanced Robotics (INFR112132022) software labs

These instructions are written for ARO labs regarding set up on DICE environment.

The lab instructions are given in the instructions notebook. 
This readme provides you with the instructions for installing the lab requirements.
**These instructions assume that you have already run the [tutorials instructions](https://github.com/ediadvancedrobotics/tutorials).** Note that the lab includes an additional dependency to the pybullet package. 

## Set up - Python 3.11

### On a DICE machine
On DICE, we will clone the [lab repository](https://github.com/ediadvancedrobotics/lab) and install the required [dependencies](https://github.com/ediadvancedrobotics/lab/blob/main/requirements.txt). 
You can "clone" the project to a local folder of your choice.
Open a terminal (CTRL + ALT + T) and follow the commands below:

-   Move to home directory.

```bash
cd ~
```
  
-   Create the aro directory if not already done

```bash
mkdir -p aro && cd aro
```

- Clone the lab inside your home directory and cd into the folder

```bash 
git clone https://github.com/ediadvancedrobotics/lab/ && cd lab
```

- Install dependencies

```bash
cd lab
conda env update -f environment.yml
conda activate aro2026
```    

Activating the environment also puts `meshcat-server` on PATH; no `.bashrc` edits
are needed.


You should be done! See [below](#using-and-updating-the-notebooks) to check that your installation is working 

### Linux, Python 3, PyPI

On a Linux system with Python 3.11 and PyBullet already installed (for example via
the Conda setup above), install the remaining dependencies with +[pip (see installation procedure and update below)](#installing-pip):
```bash
python3 -m pip install -r requirements.txt
```

Once you have the dependencies, you can start the server with `jupyter notebook .`

## Using and updating the repository
You **must** create [a local fork](https://docs.github.com/en/pull-requests/how-tos/work-with-forks/fork-a-repo) of the repository on your github account to be able to save and commit your changes to the project.

### Running the instructions notebook
On your terminal, cd into the lab folder:
```bash
cd  ~/aro/lab/
```
Now run Jupyter notebook with the command
```bash
jupyter notebook .
```
Click on 'instructions.ipynb ' to open the instructions notebook.


### Other helpful instructions
There is a pinocchio cheat sheet available as a pdf. You can also run the notebook "A_pinocchio_cheat_notebook.ipynb" to get a summary of the instructions.
Pinocchio is a bit dense and has its own singular API, it might take some time for you to become familiar with it, but trust me, this will prove largely beneficial.

### Editing the notebook and updates
If the repository changes (for example if a bug has been found and corrected by the ARO staff), you will need to update your local
version by "pulling" it from the repository. On a native installation, just go in the folder containing the lab and execute ```git pull```


## Side notes

### Installing pip

Pip is a tool for installing and managing Python packages. You can install it with

```bash
sudo apt install python3-pip
```

The default version of +pip installed by +apt is not up to date, so upgrade it with
```bash
python3 -m pip install --upgrade --user
```

In general, running +pip is likely to run an alias on +pip in /usr, so either run it through python3 as explained above, or make sure your path select the right pip executable in your ~/.local. The option --user is kind of optional for recent +pip version, but removing it should work with a warning.
