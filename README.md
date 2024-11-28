# HCL and general CS research skills 

Getting started:

1. start the GPU instance on AWS 

2. download and install XQuartz (pkg) on your mac, ssh with X forwarding into the GPU instance using public IPv4 address and the .pem key sent to you, e.g. 
ssh -i key.pem -X ubuntu@52.10.30.80

note: X forwarding makes it possible to use UI on the AWS server, e.g. emacs, or matplotlib plots can pop up in an X window. If you use mac as the ssh client, install XQuartz to enable X forwarding.
note: X fowarding doesn't work with other matplotlib uses other backend e.g. TkAgg or Qt5Agg, just use pyplot.savefig() for now

3. after ssh into the machine, run
screen
to start a screen session, google screen to get familiar with the tool. screen is like 'windows for linux' each screen is a shell command line and can run multiple commands in parallel if they are in different screens. This is the basics of getting things done on a remote server. know: 
- how to get back to the non-screen shell
- how to then, return to the screen (screen -rda)
- how to scroll inside a screen 
- how to list the screen sessions (screen -ls)
- how to kill a screen, in the screen and out of it 
- how to create more sub-screens in one screen, and list them (ctrl a ")
- how to switch across the different sub-screens (ctrl a a) 

4. activate the python virtual environment inside the screen, and whenever you want to run the code for this project.
cd code/HCL
source hcl/bin/activate

note a virtual environment is specific to python, it contains all the dependencies a project needs, but doesn't interfer with other virtual environment and the projects in other environments. So when the venv is activated, and you install packages e.g. using pip3, it stays inside this venv. It also has a list of packages installed, along with other version control or package capabilities, which you can research and look into. 

5. use emacs or vim, below are specific to emacs: 
- how to start with and without UI intervace 
- how to open a file with a buffer 
- how to naviate, end of line, start of line, start of file, end of file, page up page down 
- cut lines (ctrl k), and setting anchors (ctrl space) and select/cut 
- how to search and navigate efficiently (ctrl s) 
- save and close 
- how to search and replace a string 
- how to split the screen on a terminal and open different files 

Others: 
get familiar with git - i think you're doing well here 


