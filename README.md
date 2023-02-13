# VSCode_tutorial
A tutorial on how to use VSCode

## VSCode basics

> The following general introduction is entirely generated by ChatGPT!

Visual Studio Code (VSCode) is a popular, open-source code editor developed by Microsoft. It is widely used by developers for writing, debugging, and testing code across various programming languages and platforms. VSCode is known for its user-friendly interface, fast performance, and robust set of features.

VSCode is highly customizable and extensible, allowing developers to install extensions that enhance its functionality. These extensions can range from adding syntax highlighting for a new programming language to providing integrated debugging support for a specific framework. VSCode also supports source control systems such as Git, making it easy for developers to collaborate and manage their codebase.

Additionally, VSCode is lightweight and fast, making it a popular choice among developers who need to work on large projects or who have limited resources on their machine. It is available on Windows, Mac, and Linux platforms and can be easily installed and configured to meet the specific needs of each developer.

Whether you are a beginner or an experienced developer, VSCode is a powerful tool that can help you write, debug, and test code faster and more efficiently.

### Official documentation & tutorial

VSCode is very well documented and you can find the official documentation here: https://code.visualstudio.com/docs.

In this tutorial, I am assuming that you have actually gone through some of the basics in the official documentation website.

### Workspace

> Intro generated by ChatGPT:

One of the key features of VSCode is the concept of workspaces. In VSCode, a workspace is a virtual environment where you can organize, manage, and access all the related files, folders, and configuration settings required for your project.

A workspace in VSCode allows you to save the state of your project and its associated files, so that you can quickly return to the same state and work environment in the future. It provides a way to save your customized settings, extensions, and workspace-specific configurations, making it easier for you to switch between projects and keep your development environment organized.

You can create a workspace in VSCode by opening a folder in the editor and saving it as a workspace. You can then add other folders, files, and settings to this workspace to create a complete development environment tailored to your needs. The workspaces in VSCode can be saved and shared, allowing teams to collaborate on projects and maintain a consistent development environment.

**You will need to load the entire folder of this repository into VSCode and use it as your workspace for this tutorial.**

### Language servers

> Introduction from ChatGPT:

A language server is a program that provides language-specific services to development tools, such as IntelliSense, code navigation, and diagnostics. In Visual Studio Code (VSCode), a language server protocol (LSP) is used to connect the editor to the language server.

With a language server, you can access a wide range of language-specific features, including:

- Code completions and suggestions: As you type, the language server can suggest completions and provide documentation for variables, functions, and other language constructs.

- Code navigation: Jump to the definition of variables, functions, and other symbols with a single click.

- Diagnostics: Receive real-time feedback on potential issues in your code, such as syntax errors and type mismatches.

- Formatting: Automatically format your code as you type or on demand, according to the conventions of the language you are using.

- Refactoring: Make large-scale changes to your code, such as renaming variables and extracting methods, with just a few clicks.

In VSCode, you can install language servers for a variety of programming languages, including Java, Python, JavaScript, and many others. To use a language server, simply install the corresponding extension for that language and start writing code. The language server will automatically start providing you with the features described above.

Have a look at this webpage discussing [Programmatic Language Features](https://code.visualstudio.com/api/language-extensions/programmatic-language-features).

### Extensions

> ChatGPT:

Extensions in Visual Studio Code (VSCode) allow customization and enhancement of the editor's functionality. The VSCode marketplace has a vast array of extensions, covering various topics such as language support, debugging, code formatting, and more. Installing, enabling, and updating extensions is easy and straightforward. These extensions play a crucial role in making VSCode a flexible and adaptable development environment.

Below, I list some of the extensions I use on a daily basis:

#### Vim extension

I have used Emacs for 10+ years and I have adopted the Vim-style keybindings in Emacs since I started my PhD. I like it as I think it can significantly improve your efficiency in daily programing tasks. Feel free to learn a bit more about it.

> Again, from ChatGPT:

The Vim extension for Visual Studio Code (VSCode) is a popular plugin that brings the power and versatility of the Vim text editor to the VSCode environment. The Vim extension is designed for developers who are familiar with Vim and want to continue using its commands and features within the VSCode interface.

With the Vim extension, you can use Vim key bindings and commands in VSCode, making it easier for you to navigate and edit text in the editor. The extension provides a full implementation of Vim, including normal mode, insert mode, and visual mode, so you can use all of Vim's powerful text manipulation commands within VSCode.

In addition to the Vim commands and key bindings, the Vim extension also provides a number of configuration options that allow you to customize the behavior of the extension to suit your needs. For example, you can change the mapping of Vim keys to VSCode commands, customize the colors and styling of the Vim interface, and configure other settings such as cursor behavior and text selection.

Whether you are a Vim veteran or a newcomer looking to learn its powerful commands, the Vim extension for VSCode provides a convenient and powerful way to integrate the features of Vim into your development workflow.

#### [GitHub Copilot](https://marketplace.visualstudio.com/items?itemName=GitHub.copilot)

I like this extension and I am actually pay for it. It prompts me to write way more comments in my code now because it simply would not work without somehow detailed comments. It's way smarter than what I expect most of the times although it does give me laughable suggestions all the time. You just need to learn how you can make it work better for your needs.

> Introduction from ChatGPT (The tech behind this extension is actually developed by OpenAI, the company which has also developed ChatGPT):

GitHub Copilot is an AI-powered code suggestion tool developed by GitHub. It uses machine learning to provide intelligent code suggestions and helps developers write code faster and more efficiently. Copilot integrates directly into the GitHub coding experience, providing context-aware suggestions in real-time as developers type. With Copilot, developers can focus on writing code and let the tool handle repetitive and time-consuming tasks. It's an innovative tool that helps to boost productivity and reduce time spent on coding.

#### C/C++ 

- [C/C++](https://marketplace.visualstudio.com/items?itemName=ms-vscode.cpptools) 
- [C/C++ Extension Pack](https://marketplace.visualstudio.com/items?itemName=ms-vscode.cpptools-extension-pack)
- [C/C++ Themes](https://marketplace.visualstudio.com/items?itemName=ms-vscode.cpptools-themes)

You would need this if your projects contain C/C++ codes. They are also needed for debugging, surprisingly, Fortran codes as you will see later when I talk about this.

#### [ChatGPT](https://marketplace.visualstudio.com/items?itemName=gencay.vscode-chatgpt)

You might also want this!

#### Jupyter-related extensions

- [Jupyter](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter)
- [Jupyter Cell Tags](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.vscode-jupyter-cell-tags)
- [Jupyter Notebook Renderers](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter-renderers)
- [Jupyter Slide Show](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.vscode-jupyter-slideshow)

#### [Latex Workshop](https://marketplace.visualstudio.com/items?itemName=James-Yu.latex-workshop)

If you write LaTex, then you probably also need this. I am only using this on Linux (or Mac) as I have never figured out how to get LaTex work on Windows. You may want to have a look at [WSL](https://learn.microsoft.com/en-us/windows/wsl/install) if you are working with a Windows machine.

#### Python

The must-have for Python programmers.

- [Python](https://marketplace.visualstudio.com/items?itemName=ms-python.python)
- [Pylance](https://marketplace.visualstudio.com/items?itemName=ms-python.vscode-pylance)

#### [Modern Fortran](https://marketplace.visualstudio.com/items?itemName=fortran-lang.linter-gfortran)

For the holy language of all times!

### Terminals in VSCode

> Introduction generated by ChatGPT

Working with terminals in Visual Studio Code (VSCode) can greatly enhance your productivity as a developer. Terminals allow you to run command-line tools and scripts directly from within the editor, eliminating the need to switch between the editor and the terminal window.

Here's a brief introduction on how to work with terminals in VSCode:

- Open the Terminal in VSCode by clicking on the Terminal menu in the top menu bar or press Ctrl + ``

- By default, a new terminal window will be opened at the bottom of the editor. You can open multiple terminal windows to run multiple commands at the same time.

- Use the terminal as you would in any other terminal window. For example, you can run shell commands, navigate the file system, or run scripts.

- You can change the default shell used by the terminal in the settings. To do this, go to the User Settings (Ctrl + ,) and search for terminal.integrated.shell.windows or terminal.integrated.shell.osx depending on your operating system.

- You can also customize the terminal appearance and behavior in the settings, such as changing the font size, colors, and cursor shape.

- You can easily split or delete a terminal by clicking the two buttons at the end of the title of a terminal by first hovering your cursor on the title. The title is located at the right side of the terminal (lists of opened terminals).

- You can open a terminal at a given folder by right-clicking the folder's name in the `Explorer` panel.

By using terminals in VSCode, you can save time and improve your workflow by running commands and scripts directly from the editor.

### <a name="debugger"></a>Debugger in VSCode

Debugging is an essential part of software development and allows developers to find and fix errors in their code. Visual Studio Code (VSCode) is a popular code editor that comes with a built-in debugger to make debugging a breeze.

Microsoft provides a very detailed documentation on how to debug various types of codes [here](https://code.visualstudio.com/docs/editor/debugging). However, the procedure on how to create the `launch.json` file never works for me and it took me a while to figure out what to do, especially when I am debugging Fortran codes.

Here, I provide a workaround if the above documentation does not work for you either. 

- Create a folder named `.vscode` (note the `.` before vscode indicating it's a hidden file) in the root folder of the current workspace.

- Create an empty file named `launch.json`

- Add specific configurations on how to debug your code within the `workspace`.

Below I have provided three examples on how to debug Python, non-MPI Fortran, and MPI Fortran codes.

```json
{
    // Use IntelliSense to learn about possible attributes.
    // Hover to view descriptions of existing attributes.
    // For more information, visit: https://go.microsoft.com/fwlink/?linkid=830387
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Python: Current File",
            "type": "python",
            "request": "launch",
            "cwd": "${workspaceFolder}/src/Python",
            "program": "${file}",
            "console": "integratedTerminal",
            "justMyCode": false,
        },
        {
            "name": "Total",
            "type": "cppdbg",
            "request": "launch",
            "program": "/home/xsl/work/svn/peter/trunk/exe/debug/intel/EM3D",
            "args": ["emdata.inp"], // Possible input args for a.out
            "stopAtEntry": false,
            "cwd": "/home/xsl/writing/Papers/Total_modeling/layered_with_discs/input",
            "environment": [],
            "externalConsole": false,
            "MIMode": "gdb",
            "MIDebuggerPath": "/opt/intel/oneapi/debugger/latest/gdb/intel64/bin/gdb-oneapi",
            "justMyCode": false,
            "setupCommands": [
            {
                "description": "Enable pretty-printing for gdb",
                "text": "-enable-pretty-printing",
                "ignoreFailures": true
            }
            ]
        },
        {
            "name": "EM3D_MPI",
            "type": "cppdbg",
            "request": "attach",
            "processID": "${command:pickProcess}",
            "program": "/home/xsl/work/svn/peter/trunk/exe/debug_MPI/intel/EM3D",
            "args": ["emdata.inp"], // Possible input args for a.out
            "stopAtEntry": false,
            "cwd": "/home/xsl/writing/Papers/Perfect_conductors/Ovoid_tetgen/m1_ramp_off/m1/input/",
            "environment": [],
            "externalConsole": false,
            "MIMode": "gdb",
            "MIDebuggerPath": "/opt/intel/oneapi/debugger/latest/gdb/intel64/bin/gdb-oneapi",
            "justMyCode": false,
            "setupCommands": [
            {
                "description": "Enable pretty-printing for gdb",
                "text": "-enable-pretty-printing",
                "ignoreFailures": true
            }
            ]
        },
    ]
}
```

### Developing Python codes using Jupyter Notebook

#### Jupyter Notebook extension

- Developing Python codes with the Jupyter Notebook extension in probably very similar to how you do it using the traditional web-based Jupyter Notebook (writing and execution). 

- You can also debug the code in each of the code cells using the VSCode debugger (you would probably need the [JupyterLab](https://jupyter.org/install) software for debugging).

- In-line (interactive) images can be rendered normally.

- You can also export the Jupyter Notebook to a normal Python script (.py) and save it.

    Click `...` at the end of the toolbar and select `Export` and save it as `Python Script`.

#### Normal Python scripts

- You can certainly write *.py files using VSCode.

- Full support of the Python language server provide by [Pylance](https://marketplace.visualstudio.com/items?itemName=ms-python.vscode-pylance).

- Running your code interactively

- Running your code in terminal

- Jupyter integration (e.g., showing variable names)

- Debugger

### Developing Fortran codes

#### [fortls](https://github.com/fortran-lang/fortls)

The language server required by the `Modern Fortran` extension should be installed first. Without `fortls`, the extension `Modern Fortran` would not work properly. Please refer to the official GitHub page for details on how to install `fortls`.

#### Modern Fortran extension settings

There are a few places you need to set up before `Modern Fortran` can work properly. To change the settings, simply open the command pellet (`Ctrl+Shift+P`) and type `settings`. In the search box, type in `Fortran` and then you will see all settings related to `Modern Fortran`. Alternatively, you can also click the settings button in the extension panel once you locate `Modern Fortran`.

You can apply the changes to settings at various levels. I typically only set the `Remote` level (WSL) as that's where I develop my codes. You can also have unique settings for a single workspace, etc.

- General

    Select `fortls` for `Autocomplete`, `Hover`, and `Symbols`. For perferred case, select according to your own preferences.

- Linter

> Introduction from ChatGPT:

> A linter is a tool that analyzes code for potential errors and helps to enforce a coding standard. It scans the codebase for issues such as syntax errors, code style inconsistencies, and logical problems, and provides warnings and suggestions for how to correct them. Linters are typically used in software development to help ensure that code is written in a consistent and maintainable way, and to catch potential issues before they become bigger problems.

> There are linters available for many programming languages, including JavaScript, Python, Ruby, and many others. Some linters are integrated into code editors, making it easy for developers to run them as they write code. Other linters are run as part of a continuous integration and deployment process, checking code automatically whenever changes are made.

Linter for Fortran is provided by the `Modern Fortran` extension and it relies on a Fortran compiler to work. There are a few settings related to linter and you need to set it up based on your own preferences.

1. Fortran > Linter > Compiler: Select the compiler you want to use, e.g., `ifort` or `gfortran`. They have to be available on your computer.

2. Fortran > Linter > Compiler Path: I use `ifort` and it is located at `/opt/intel/oneapi/compiler/2023.0.0/linux/bin/intel64/ifort` on my computer. If the compiler is actually in your `PATH` [environment variable](https://en.wikipedia.org/wiki/Environment_variable), then you probably do not need to specify the path here but I am not entirely sure.

3. Fortran > Linter > Include Paths: This is where you set the path of the directory containing your `.mod` files are stored. Most likely, your Fortran project contains more than one `.f90` file and if you do not set this path, anything from other `.f90` files would be unrecognizable from the current file you are working on. It's a good practice to keep the `.mod` files in a separate directory from the directory where all the source files reside. For example, most commonly, people would created a `src` folder designated for `.f90` files and a `build` folder for `.mod` and `.o` files.

4. Fortran > Linter > Mod Output: Tha path to the directory where you temporarily store a `.mod` file generated by the compiler whenever a `.f90` file is opened. The linter invokes the compiler specified above and compiles the `.f90` file as you edit. A temporary `.mod` file will be generated after the compilation, and the directory set up here is used to store that `.mod` file. Also **remember** to put this directory in the above `Include Paths` setting because you may be editing multiple files simultaneously.

- Formatting

1. Fortran > Formatting > Formatter: Set up the software you will use to format your Fortran codes. There are two options: [fprettify](https://github.com/pseewald/fprettify) and [findent](https://pypi.org/project/findent/). Both of them can be installed by using `Pip`. For example, you can install `fprettify` by typing `pip install fprettify` in your terminal. The default installation directory is `~/.local/bin`.

2. Fortran > Formatting > Path: The directory where the formatting library resides (e.g., `~/.local/bin`). It would not be visible to `Modern Fortran` if you do not set this up.

- Language Server

1. Fortran > Fortls > Path: The directory which contains the `fortls` program. As I installed `fortls` using `pip`, its `~/.local/bin/fortls` for me. Note, you probably have to put `fortls` after `~/.local/bin` as what I did, although this is totally against my understanding of what `Path` is. Give it a try and see which one works for you.

2. Fortran > Fortls > Directories: The directories where you have your `.f90` files for your projects. `Fortls` require this so that it can read all the source files to work properly.

3. Fortran > Fortls > Preprocessor: If you have any preprocessor files, then you need to set this one as well. If you do not know what is a preprocessor file, then chances are you do not need this.

#### Debugging sequential or shared-memory (non-MPI) parallel Fortran codes

Debugging sequential or shared-memory parallel Fortran codes is easy. After compilation and having set up the `launch.json` file ([see here](#debugger-in-vscode)), all you need to do is follow the standard procedure as detailed in the official documentation given [here](https://code.visualstudio.com/docs/editor/debugging).

#### Debugging distributed-memory (MPI) parallel Fortran codes

Debugging distributed-memory parallel codes in Fortran is a bit more complex. We need to run the MPI program first in terminal and then attach the debugger to the processes spawned by the MPI program. Please have a look at this [page (item 6)](https://www.open-mpi.org/faq/?category=debugging) from OpenMPI for a detailed explanation why you would need to do this.

Each VSCode window can only handle a single process so that you may need to open multiple VSCode windows. You cannot open the same workspace twice (VSCode would point you to the already opened window). I would simply open a new window and then add the folders to the workspace that won't be saved. Also, try to limit the number of MPI processes you use to launch your MPI program. Sometimes you would need to control how the program runs in each process. If you use too many processes then you would need to open many VSCode windows!

[Here](https://iamsorush.com/posts/debug-mpi-vs-code/) is a good tutorial on how to debug MPI codes using VSCode (using C++ as an example). All we need to pay attention to is how to exit the sleep function which I have provided a Fortran version in the file `toy_mpi.f90` in the `src` directory.

```fortran {.line-numbers}
SUBROUTINE MPI_debug_vscode(myid)

    !! When debugging with VSCode, you need to manually change the value of the variable ii from 
    !! 0 to 1 in order to continue running the program
    !! We assume that MPI_init has been called

    INTEGER, INTENT(IN) :: myid
    INTEGER :: ii, ierr

    PRINT*, 'RANK: ', myid

    ii = 0
    IF (myid == 0)THEN
        DO WHILE (ii /= 1)
        ! We only execute this on the master process so that we do not need to fire up
        ! vscode instances for all processes to step out the loop
        CALL fortran_sleep(3)
        END DO
    END IF
    ! A barrier is required so that all processes will stay at this point while the 
    ! master process is being attached (or any other slave processes are being 
    ! attached when needed)
    CALL MPI_Barrier(MPI_COMM_WORLD, ierr)

END SUBROUTINE MPI_debug_vscode
```