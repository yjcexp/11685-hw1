git status
git add .
git commit -m "Describe what you changed"
git push



• Autograde your code by
– Step 1: Open your preferred IDE or code editor, locate and open the desired .py file, make the
necessary edits, and save the changes. Ideally, the homework does not require GPU usage.
– Step 2: (IMPORTANT): Setting the flags in hw1p1 autograder flags.py to True to test any individual component on your local autograder. For example, if you only implement the sigmoid activation functions, set DEBUG AND GRADE SIGMOID flag = True and everything else to False.
– Step 3: Running local autograder by: Confirm that you are the top-level directory and execute the following in anaconda prompt or terminal:
                   (/hw1p1_handout/) $ python3 autograder/hw1p1_autograder.py
It is recommended to set up a new local environment and install the library dependencies versions for the homework to avoid version compatibility problems.
Please remember that the local autograder only has a few tests as a preliminary check. The entire suite of tests is run on Autolab after you hand-in your code as described below.
• Hand-in your code by running the following command from the top level directory, then SUBMIT the created handin.tar file to autolab3 (please let the team know if you face an issue in the TARing step):
               (/hw1p1_handout/) $ tar -cvf handin.tar models mytorch
Note: After the Tar operation, to ensure your Tar process is done correctly. You can Untar the Tar file, and your Untar folder must contain the models and mytorch subfolders.
• DO
– Make sure you understand the concept of each function, we don’t want you to ”translate” math equations to codes without understanding them.
– Go through the examples we provide to have a better visualization of the matrix calculations. If you ask TAs for help, we will ask you to explain the example to us before giving you more hints.