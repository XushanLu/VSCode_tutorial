# VSCode_tutorial

A tutorial on how to use VSCode mainly for research-related coding. This is prepared for my live demo for the department. So, something may be missing here because I will demonstrate during the presentation.

## Motivation

### TL, DR

- Expose VSCode to those of you who are only using basic text editors or web-based Jupyter Notebook
- Show how a tool like VSCode can potentially increase productivity
- Show how to debug your code (Python, Fortran, etc.) using an IDE instead of print debugging
- Show how version control can be useful for your coding and research in general
- Show how AI (mostly GitHub CoPilot) can be used to turbo-charge your coding productivity

### A bit more in detail

Over the years, I have seen people struggling using very basic tools in their research, not being exposed to other better tools such as VSCode. In the past, I've mainly used Emacs (specifically, [Spacemacs](https://www.spacemacs.org/)) for anything related to text editing (e.g., writing papers, coding for research). In fact, with crazy projects like [Emacs-application-framework](https://github.com/emacs-eaf/emacs-application-framework), you can literally 'live' in Emacs (i.e., getting everything done inside Emacs, no terminals, no PDF viewers, no internet browsers, etc.). But learning Emacs is difficult, and mastering it requires a good knowledge of [Elisp](https://en.wikipedia.org/wiki/Emacs_Lisp), the language behind the Emacs text editor. If you don't know that language well, you cannot really configure your Emacs, except for copying others' configurations online and adjust it bit by bit from trial-and-error. And also there's the [Vim](https://www.vim.org/) editor which I actually don't know much about except for it's keybindings (as I use it with, strangely, Emacs, to avoid the terrible [Emacs Pinky](http://xahlee.info/emacs/emacs/emacs_pinky.html) problem). If you actually clicked the links I gave above, you will realize one thing: the websites all seem to be from last century. You are not wrong! Most of the people who are still using these editors learned how to use a computer in the last century. And they spent too much time waging [editor wars](https://en.wikipedia.org/wiki/Editor_war) on each other so they don't have much time to make their website up to date in terms of how they look.

I started coding seriously at the very end of my undergrad years (2011-2012) using Intel Visual Fortran and Microsoft's [Visual Studio](https://visualstudio.microsoft.com/). Before that I learned C in the first year of my undergrad and I remember we were using Microsoft's [Visual C++ (version 6.0)](https://en.wikipedia.org/wiki/Microsoft_Visual_C%2B%2B). I quickly stopped using it once I started my master's program and switched from Windows to Linux (I discovered Emacs!). At the time, probably none of the compiler nor the (integrated development environment) IDE was free, and they were very heavyweight. When I started using Emacs, I find it's possible to write my code much faster than what I could do with Visual Studio's default editor. I quickly became used to keyboard-driven editing where you don't actually need to switch between your keyboard and your mouse, which becomes extremely annoying once you are used to keyboard-driven editing (see [here](https://waylonwalker.com/keyboard-driven-vscode/) to find out how far you can go along the keyboard-driven editing even with VSCode). But one thing I did find missing after the switch was the powerful debugging experience offered by Visual Studio. I had to rely on [print debugging](https://en.wikipedia.org/wiki/Debugging#Techniques). I asked ChatGPT whether print debugging is good and here's the answer:

> Print debugging can be a useful technique for debugging code, particularly for small scripts or when working in environments that do not support more advanced debugging tools. By adding print statements throughout your code, you can print out the value of variables, track the execution flow of your code, and identify where errors occur.

> However, print debugging can be a tedious and time-consuming process, particularly for larger codebases. It can also be difficult to trace errors that occur within loops or conditionals that are nested deeply within your code.

> In general, print debugging can be a good starting point for debugging your code, but it should not be relied upon as the only means of debugging. If you are working on a larger codebase or need to debug more complex issues, you may want to consider using more advanced debugging tools, such as debuggers, profilers, or logging frameworks, which can provide more detailed insights into your code's execution and make it easier to identify and fix errors.

So, it would be wonderful if there are tools that can provide a smooth editing experience and serve as an IDE at the same time. Also, it should not be too difficult to learn as Emacs and Vim. I think VSCode is probably the answer.

Below I provided some information about VSCode and what you can use it to do. I will simply go through all of them during the live demo.

## Prerequisite for this tutorial

There are some tools you need to install before going through this tutorial. Of course, you have to have VSCode installed on your computer at least. Below, I list some other tools that I can think of.

- Windows Subsystem for Linux
    For Windows users, I strongly recommend installing [Windows Subsystem for Linux](https://learn.microsoft.com/en-us/windows/wsl/install). This tutorial assumes you are working with a Linux OS. Everything probably still apply if you are working on a Mac, but I have no experience of installing all necessary tools on a Windows machine and make it work.

- Python interpreter
    You can use the interpreter that comes with your system, or you can install [Anaconda](https://www.anaconda.com/). I use the interpreter provided by Intel (more below).

- Fortran Compilers (and `make`)

    If your research involves coding in Fortran, then you need to have a compiler installed to compile Fortran codes.

  - Intel compilers
    Please follow the instructions [here](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html?operatingsystem=linux&distributions=aptpackagemanager) to install the compilers on your computer. Necessary toolkits include:

    1. intel-basekit
    2. intel-hpckit
    3. intel-aikit (includes Python interpreter)

  - GNU compilers (gcc, gfortran)

    On Ubuntu, the GNU tools can be installed via `sudo apt install gfortran build-essential`. `Make` is included in the `build-essential` packages.

## VSCode basics

> The following general introduction is entirely generated by ChatGPT!

> Visual Studio Code (VSCode) is a popular, open-source code editor developed by Microsoft. It is widely used by developers for writing, debugging, and testing code across various programming languages and platforms. VSCode is known for its user-friendly interface, fast performance, and robust set of features.

> VSCode is highly customizable and extensible, allowing developers to install extensions that enhance its functionality. These extensions can range from adding syntax highlighting for a new programming language to providing integrated debugging support for a specific framework. VSCode also supports source control systems such as Git, making it easy for developers to collaborate and manage their codebase.

> Additionally, VSCode is lightweight and fast, making it a popular choice among developers who need to work on large projects or who have limited resources on their machine. It is available on Windows, Mac, and Linux platforms and can be easily installed and configured to meet the specific needs of each developer.

Whether you are a beginner or an experienced developer, VSCode is a powerful tool that can help you write, debug, and test code faster and more efficiently.

### Official documentation & tutorial

VSCode is very well documented and you can find the official documentation here: https://code.visualstudio.com/docs.

In this tutorial, I am assuming that you have actually gone through some of the basics in the official documentation website. If you don't have enough time, then at least you should have a look at the [Tips and Tricks](https://code.visualstudio.com/docs/getstarted/tips-and-tricks) page which shows off a broad range of fancy stuff VSCode can do.

### Workspace

> Intro generated by ChatGPT:

> One of the key features of VSCode is the concept of workspaces. In VSCode, a workspace is a virtual environment where you can organize, manage, and access all the related files, folders, and configuration settings required for your project.

> A workspace in VSCode allows you to save the state of your project and its associated files, so that you can quickly return to the same state and work environment in the future. It provides a way to save your customized settings, extensions, and workspace-specific configurations, making it easier for you to switch between projects and keep your development environment organized.

> You can create a workspace in VSCode by opening a folder in the editor and saving it as a workspace. You can then add other folders, files, and settings to this workspace to create a complete development environment tailored to your needs. The workspaces in VSCode can be saved and shared, allowing teams to collaborate on projects and maintain a consistent development environment.

**You will need to load the entire folder of this repository into VSCode and use it as your workspace for this tutorial.**

### Language servers

> Introduction from ChatGPT:

> A language server is a program that provides language-specific services to development tools, such as IntelliSense, code navigation, and diagnostics. In Visual Studio Code (VSCode), a language server protocol (LSP) is used to connect the editor to the language server.

> With a language server, you can access a wide range of language-specific features, including:

> - Code completions and suggestions: As you type, the language server can suggest completions and provide documentation for variables, functions, and other language constructs.

> - Code navigation: Jump to the definition of variables, functions, and other symbols with a single click.

> - Diagnostics: Receive real-time feedback on potential issues in your code, such as syntax errors and type mismatches.

> - Formatting: Automatically format your code as you type or on demand, according to the conventions of the language you are using.

> - Refactoring: Make large-scale changes to your code, such as renaming variables and extracting methods, with just a few clicks.

> In VSCode, you can install language servers for a variety of programming languages, including Java, Python, JavaScript, and many others. To use a language server, simply install the corresponding extension for that language and start writing code. The language server will automatically start providing you with the features described above.

Have a look at this webpage discussing [Programmatic Language Features](https://code.visualstudio.com/api/language-extensions/programmatic-language-features).

### Extensions

> ChatGPT:

> Extensions in Visual Studio Code (VSCode) allow customization and enhancement of the editor's functionality. The VSCode marketplace has a vast array of extensions, covering various topics such as language support, debugging, code formatting, and more. Installing, enabling, and updating extensions is easy and straightforward. These extensions play a crucial role in making VSCode a flexible and adaptable development environment.

Below, I list some of the extensions I use on a daily basis:

#### Vim extension

I have used Emacs for 10+ years and I have adopted the Vim-style keybindings in Emacs since I started my PhD. I like it as I think it can significantly improve your efficiency in daily programing tasks. Feel free to learn a bit more about it.

> Again, from ChatGPT:

> The Vim extension for Visual Studio Code (VSCode) is a popular plugin that brings the power and versatility of the Vim text editor to the VSCode environment. The Vim extension is designed for developers who are familiar with Vim and want to continue using its commands and features within the VSCode interface.

> With the Vim extension, you can use Vim key bindings and commands in VSCode, making it easier for you to navigate and edit text in the editor. The extension provides a full implementation of Vim, including normal mode, insert mode, and visual mode, so you can use all of Vim's powerful text manipulation commands within VSCode.

> In addition to the Vim commands and key bindings, the Vim extension also provides a number of configuration options that allow you to customize the behavior of the extension to suit your needs. For example, you can change the mapping of Vim keys to VSCode commands, customize the colors and styling of the Vim interface, and configure other settings such as cursor behavior and text selection.

> Whether you are a Vim veteran or a newcomer looking to learn its powerful commands, the Vim extension for VSCode provides a convenient and powerful way to integrate the features of Vim into your development workflow.

#### [GitHub Copilot](https://marketplace.visualstudio.com/items?itemName=GitHub.copilot)

I like this extension and I am actually paying for it. It prompts me to write way more comments in my code now because it simply would not work without somehow detailed comments. It's way smarter than what I expect most of the times although it does give me laughable suggestions all the time. You just need to learn how you can make it work better for your needs.

> Introduction from ChatGPT (The tech behind this extension is actually developed by OpenAI, the company which has also developed ChatGPT):

> GitHub Copilot is an AI-powered code suggestion tool developed by GitHub. It uses machine learning to provide intelligent code suggestions and helps developers write code faster and more efficiently. Copilot integrates directly into the GitHub coding experience, providing context-aware suggestions in real-time as developers type. With Copilot, developers can focus on writing code and let the tool handle repetitive and time-consuming tasks. It's an innovative tool that helps to boost productivity and reduce time spent on coding.

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

## Terminals in VSCode

> Introduction generated by ChatGPT

> Working with terminals in Visual Studio Code (VSCode) can greatly enhance your productivity as a developer. Terminals allow you to run command-line tools and scripts directly from within the editor, eliminating the need to switch between the editor and the terminal window.

> Here's a brief introduction on how to work with terminals in VSCode:

> - Open the Terminal in VSCode by clicking on the Terminal menu in the top menu bar or press Ctrl + ``

> - By default, a new terminal window will be opened at the bottom of the editor. You can open multiple terminal windows to run multiple commands at the same time.

> - Use the terminal as you would in any other terminal window. For example, you can run shell commands, navigate the file system, or run scripts.

> - You can change the default shell used by the terminal in the settings. To do this, go to the User Settings (Ctrl + ,) and search for terminal.integrated.shell.windows or terminal.integrated.shell.osx depending on your operating system.

> - You can also customize the terminal appearance and behavior in the settings, such as changing the font size, colors, and cursor shape.

> - You can easily split or delete a terminal by clicking the two buttons at the end of the title of a terminal by first hovering your cursor on the title. The title is located at the right side of the terminal (lists of opened terminals).

> - You can open a terminal at a given folder by right-clicking the folder's name in the `Explorer` panel.

> By using terminals in VSCode, you can save time and improve your workflow by running commands and scripts directly from the editor.

## Debugger in VSCode

> From ChatGPT

> Debugging is an essential part of software development and allows developers to find and fix errors in their code. Visual Studio Code (VSCode) is a popular code editor that comes with a built-in debugger to make debugging a breeze.

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
            "name": "toy",
            "type": "cppdbg",
            "request": "launch",
            "program": "${workspaceFolder}/build/a.out",
            "args": [], // Possible input args for a.out
            "stopAtEntry": false,
            "cwd": "${workspaceFolder}/build",
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
            "name": "toy_MPI",
            "type": "cppdbg",
            "request": "attach",
            "processID": "${command:pickProcess}",
            "program": "${workspaceFolder}/build/b.out",
            "args": [], // Possible input args for a.out
            "stopAtEntry": false,
            "cwd": "${workspaceFolder}/build",
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

## Developing Python codes using Jupyter Notebook

### Jupyter Notebook extension

- Developing Python codes with the Jupyter Notebook extension in probably very similar to how you do it using the traditional web-based Jupyter Notebook (writing and execution). 

- You can also debug the code in each of the code cells using the VSCode debugger (you would probably need the [JupyterLab](https://jupyter.org/install) software for debugging).

- In-line (interactive) images can be rendered normally.

- You can also export the Jupyter Notebook to a normal Python script (.py) and save it.

    Click `...` at the end of the toolbar and select `Export` and save it as `Python Script`.

- You can directly debug the Jupyter cells by clicking the `˅` button right next to the `▷` button at the top-left conner of a code cell.

### Normal Python scripts

- You can certainly write *.py files using VSCode.

- Full support of the Python language server provide by [Pylance](https://marketplace.visualstudio.com/items?itemName=ms-python.vscode-pylance).

- Running your code interactively

- Running your code in terminal

- Jupyter integration (e.g., showing variable names)

- Debugger
  - You can use the [Run and Debug](https://code.visualstudio.com/docs/editor/debugging#_run-and-debug-view) from the [Activity Bar](https://code.visualstudio.com/docs/getstarted/userinterface), or you can directly access the debugger from the tool bar located at the very end of the tab titles within the opened [Editor Group](https://code.visualstudio.com/docs/getstarted/userinterface#_editor-groups) (click the `˅` button besides `▷`).

## Developing Fortran codes

### [fortls](https://github.com/fortran-lang/fortls)

The language server required by the `Modern Fortran` extension should be installed first. Without `fortls`, the extension `Modern Fortran` would not work properly. Please refer to the official GitHub page for details on how to install `fortls`.

### Modern Fortran extension settings

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

### Compilation with a Makefile

It's practically impossible to compile a large Fortran project with dozens of source files using command line. Writing a [Makefile](https://www.gnu.org/software/make/manual/html_node/Introduction.html) and then compile the project using [GNU make](https://www.gnu.org/software/make/) simplifies the process. Each time you want to (re)compile your project, you just need to run the `make` command followed by the target name in your terminal.

In this tutorial, I have also provided a simple [Makefile](Makefile) that you can use to compile the two toy examples. To compile the file `/src/Fortran/toy.f90`, simply type `make a.out` in your terminal. To compile the MPI toy example, type `make b.out`. You would need to have the Intel compilers installed (details given [above](#prerequisite-for-this-tutorial)).

> Tricks: when running make in terminal, and when you see an error, you can quickly jump to the line caused the compilation error by holding `Ctrl` and then click that filename (with a line number) emitted by the terminal. This is much faster than opening the file and jumping to the offending line manually.

### Debugging sequential or shared-memory (non-MPI) parallel Fortran codes

Debugging sequential or shared-memory parallel Fortran codes is easy. After compilation and having set up the `launch.json` file ([see here](#debugger-in-vscode)), all you need to do is follow the standard procedure as detailed in the official documentation given [here](https://code.visualstudio.com/docs/editor/debugging).

### Debugging distributed-memory (MPI) parallel Fortran codes

Debugging distributed-memory parallel codes in Fortran is a bit more complex. We need to run the MPI program first in terminal and then attach the debugger to the processes spawned by the MPI program. Please have a look at this [page (item 6)](https://www.open-mpi.org/faq/?category=debugging) from OpenMPI for a detailed explanation why you would need to do this.

Each VSCode window can only handle a single process so that you may need to open multiple VSCode windows. You cannot open the same workspace twice (VSCode would point you to the already opened window). I would simply open a new window and then add the folders to the workspace that won't be saved. Also, try to limit the number of MPI processes you use to launch your MPI program. Sometimes you would need to control how the program runs in each process. If you use too many processes then you would need to open many VSCode windows!

[Here](https://iamsorush.com/posts/debug-mpi-vs-code/) is a good tutorial on how to debug MPI codes using VSCode (using C++ as an example). All we need to pay attention to is how to exit the sleep function which I have provided a Fortran version in the file `toy_mpi.f90` in the `src` directory. The subroutine is given below.

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

In the above subroutine, you first need to set a breakpoint at the line `CALL fortran_sleep(3)` after you have attached to the master process (the one with the lowest pid). The `launch.json` file given [above](#debugger-in-vscode) would automatically pop up a box asking you to attach the debugger to the correct process. You can simply search the pid of the program by typing the program's name in the search box. For our case here, we simply need to type `b.out`. It will show all the matching processes with a name of `b.out`. You need to select the process with the smallest number for the pid, which corresponds to the master MPI process.

After you have successfully attached the debugger to the master process, follow the tutorial given [here](https://iamsorush.com/posts/debug-mpi-vs-code/).

## Writing LaTex files

To write LaTex files, you need to use the [LaTex Workshop](https://marketplace.visualstudio.com/items?itemName=James-Yu.latex-workshop) extension. There are other similar extensions but this one is probably the most popular and the best to use. It has a very detailed [wiki page](https://github.com/James-Yu/LaTeX-Workshop/wiki) on GitHub already (note you need to click the `table of contents` on the right side of the page).

## Writing Markdown files for documentation or even to write your paper?

[Markdown](https://www.markdownguide.org/) is one of the languages called [markup languages](https://en.wikipedia.org/wiki/Markup_language). You might have already known that most of the Readme pages for GitHub repositories are written using Markdown.

I used the [Emacs Org-mode](https://orgmode.org/) a lot in the past and I even wrote my entire PhD thesis using it. I liked it very much and I still think it's a great tool. However, it's not as widely supported as [Markdown](https://www.markdownguide.org/), and you can pretty much only work with Emacs. 

Recently, I came across something called [MyST markdown tools](https://myst-tools.org/) and I am considering to use it in the future to replace Org-mode. Have a look at this [YouTube video](https://www.youtube.com/watch?v=NxSNVt9XsCI&t=1648s) and see how you can write a paper with Markdown.

Writing with LaTex only is probably something in the past.

## Version control integration in VSCode

### What is version control?

> ChatGPT:

> Version control is a system that tracks changes made to files over time and allows multiple people to work on the same project simultaneously. It is widely used in software development to manage source code, but it is also used in other fields, such as research, where it can be applied to documents, data, images, and other types of files.

> With version control, each change to a file is recorded, along with a description of the change and the person who made it. This makes it easy to track the history of a project and to revert to an earlier version if necessary. Additionally, version control systems allow multiple people to work on the same project simultaneously, by providing mechanisms for resolving conflicts when multiple changes are made to the same file.

There are several popular version control systems available, including Git, Mercurial, and Subversion. Here we only focus on Git and SVN.

### Why version control is important even for research?

> Below is the answer from ChatGPT:

> Version control is important for research for several reasons:

> - Collaboration: Research often involves collaboration with other researchers, and version control systems make it easy for multiple people to work on the same project simultaneously. With version control, each researcher can work on their own branch and merge their changes into the main branch when they are ready.

> - Traceability: Research involves a lot of experimentation, testing, and iteration. With version control, it's easy to keep track of all the changes made to a project and to revert to an earlier version if necessary. This makes it easier to understand why certain decisions were made and to replicate results.

> - Reproducibility: One of the goals of scientific research is to be able to reproduce results. With version control, it's possible to share the entire history of a project with others, including all the code, data, and documentation. This makes it easier for others to reproduce the results and build upon the work.

> - Backup: Research projects can be lost or damaged due to hardware failure, accidental deletion, or other reasons. With version control, a complete history of all changes is kept, and it's possible to recover earlier versions of the project if necessary.

> - Auditability: Some research projects, especially those in regulated industries, require a high level of accountability and transparency. Version control systems provide a clear and detailed history of all changes made to a project, making it easier to demonstrate compliance and accountability.

> In conclusion, version control is an essential tool for researchers, as it helps them to collaborate, track changes, reproduce results, backup their work, and demonstrate accountability.

### How to use version control in VSCode

VSCode integrates source control functions within the editor which means you probably do not even need to install any extension for this to work. It is designed to work with Git but it can also provides some basic supports to SVN. Please have a look at this [webpage](https://code.visualstudio.com/docs/sourcecontrol/overview) for tutorials on how to use the version control tools in VSCode.

If you are still working with SVN, then there is an extension named [SVN](https://marketplace.visualstudio.com/items?itemName=johnstoncode.svn-scm) and you can use it to track changes in the SVN repository.

If you are just starting to use version control, I recommend using Git.

## Remote development with VSCode

This is probably one of the most important reasonse why I switched from Emacs to VSCode. It's just much smoother to develop remotely with VSCode compared with Emacs. You don't really feel much difference between local and remote developments anymore. You can remotely develop your code on another machine via SSH or develop on the Windows side VSCode using your [Windows Subsystem for Linux](https://learn.microsoft.com/en-us/windows/wsl/install).

Please refer to the [official documentation](https://code.visualstudio.com/docs/remote/remote-overview) to learn how remote development works.

## Can we use AI to code?

### How the GitHub CoPilot extension is speeding up my coding productivity

#### Error check in Fortran codes

[Dr. Peter Lelièvre](https://www.esd.mun.ca/~peter/Home.html) has a large collection of Fortran codes that almost everyone of us have used in one way or another. I have been building my EM/DC/IP inversion codes on top of (mainly) his Fortran codes. He's one of the few people who really tried to do [object-oriented programming](https://en.wikipedia.org/wiki/Object-oriented_programming) using Fortran (I was heavily influenced by his code). There is one thing in his Fortran code I find very useful and I still have not found anything even remetely close to it: [error handling](https://en.wikipedia.org/wiki/Exception_handling). Unlike other languages such as Python which has built-in error handling functions, Fortran almost has nothing (well, it just does not have any meaningful standard library at all!). So, Peter coded up a Fortran module (roughly equivalent to a class as in other languages) designated to error handling in Fortran.

For example, when you check a statement and when it returns `.False.`, indicating something went wrong, then you would like to record this error and return all the way back to your main program so that you can terminate the execution of the code to figure out what's going on. Here is an example of how you construct an error object:

```fortran
! Report an error if the dimension is not 3
IF (n_dim /= 3) THEN
    CALL error_construct(error, ERROR_GENERAL, modname, subname, &
                         'The dimension must be 3');    RETURN
END IF
```

The above code constructs an error object which contains information on the type of the error `ERROR_GENERAL`, the names of the module and the subroutine in which the error occurred (used to build the [call stack](https://en.wikipedia.org/wiki/Call_stack)). However, there are lots of typing to do just this, which becomes a problem when you want to make your code really robust and add tons of error checks. But the only things that one error construction is different from another are the `if` statements and the `error message`. This is one of the most typical things that GitHub CoPilot is extremely good at! With just a brief comment (`! Report an error if the dimension is not 3`), all you need to do is typing `IF` and then you just wat for a second until all the necessary codes shows up.

There is another error construction that is slightly different from the one shown above. It is required when an `allocation` failed in Fortran for whatever reason. In this case, you need to supply `ERROR_MEMORY` as the type of the error. The error message can just be the name of the array whose allocation failed:

```fortran
! Allocate the memory for pt_ele, ln_ele, ln_ele_nodes
ALLOCATE(pt_ele(3, n_rec*4), stat=ierr)
IF (ierr /= 0) THEN
    CALL error_construct(error, ERROR_MEMORY, modname, subname, &
                         'pt_ele');    RETURN
END IF
```

After a few times, the AI bot is smart enough to figure out the difference between a normal error construction and a allocation error construction, and it then gives almost perfect codes for this. It typically does not make mistakes even you switch between the two error construction regularly. The success rate is pretty high. Now, my code has way more error check lines than before!

#### Automatically generating routines slightly different from previous ones

I am trying to generate [rotation matrices](https://en.wikipedia.org/wiki/Rotation_matrix) and I tried to make things as general as possible so that I try to write three subroutines (or functions in Python) to return a matrix in three directions and multiply them together to give me a general rotation matrix if you need to rotate things in three dimensions.

In Fortran, the rotation along the z-axis can be written as following (I basically just wrote the comments and the variable declaration and actual matrix were all accepted from the AI suggestions):

```fortran
PURE SUBROUTINE get_rotation_matrix_z(angle, mat)

    !! Return the rotation matrix for rotation around the z-axis. The positive 
    !! angle is counter clockwise when observing from the positive z-axis to the
    !! origin. This angle is simply the yaw angle in the aircraft coordinate. 
    !! See https://en.wikipedia.org/wiki/Rotation_matrix and 
    !! https://en.wikipedia.org/wiki/Aircraft_principal_axesfor more details.

    !! The angle in radians
    REAL(REAL_KIND), INTENT(IN) :: angle
    !! The rotation matrxi. Dimension (3, 3)
    REAL(REAL_KIND), INTENT(OUT) :: mat(3, 3)

    mat = reshape([COS(angle), SIN(angle), 0.0_REAL_KIND, &
                    -SIN(angle), COS(angle), 0.0_REAL_KIND, &
                    0.0_REAL_KIND, 0.0_REAL_KIND, 1.0_REAL_KIND], [3, 3])

END SUBROUTINE get_rotation_matrix_z
```

> If you compare the matrix with the one from the Wikipedia page, you will see that it fully understand how the `reshape` intrinsic function works, and it gives the 1D array in the correct order. It also knows that you cannot just use `0.0` because that will give you an error of not having the same type of data inside an array constructor. It also knows that my `REAL` variables are of the `REAL_KIND` type. Lastly, it figures out which axis corresponds to which angle (yaw, pitch, and roll)!

Then, I tried to write another one to return the matrix when you rotate things along the y-axis. The AI bot was able to generate the full subroutine after I finished typing `PURE SUBROUTINE get_rotation_matrix_y()`. I just deliberately paused a second when my cursor is inside the brackets:

```fortran
PURE SUBROUTINE get_rotation_matrix_y(angle, mat)

    !! Return the rotation matrix for rotation around the y-axis. The positive 
    !! angle is counter clockwise when observing from the positive y-axis to the
    !! origin. This angle is simply the pitch angle in the aircraft coordinate. 
    !! See https://en.wikipedia.org/wiki/Rotation_matrix and 
    !! https://en.wikipedia.org/wiki/Aircraft_principal_axesfor more details.
    
    !! The angle in radians
    REAL(REAL_KIND), INTENT(IN) :: angle
    !! The rotation matrxi. Dimension (3, 3)
    REAL(REAL_KIND), INTENT(OUT) :: mat(3, 3)

    mat = reshape([COS(angle), 0.0_REAL_KIND, -SIN(angle), &
                    0.0_REAL_KIND, 1.0_REAL_KIND, 0.0_REAL_KIND, &
                    SIN(angle), 0.0_REAL_KIND, COS(angle)], [3, 3])

END SUBROUTINE get_rotation_matrix_y
```

> I have not tested this but I have tested the Python version and this also looks like flawless to my bare eyes! (The pause is critical here! If you type too fast, the bot would not be able to respond. This just shows that it is also important to learn how to make the bot work the best for you.)

#### Writing docstring in Python

I hate writing comments, be it Fortran or Python. However, to make the language server features work (e.g., hover to see the function definition at the place where you call it), you have to have proper comments written when you write the function. For Python, you need to write proper [docstring](https://peps.python.org/pep-0257/) which can also be used to automatically generate documentation of your code with tools such as [Sphinx](https://www.sphinx-doc.org/en/master/). I find GitHub CoPilot can be used to speed up the process, especially when you name your variables with [semantic names](https://neonira.github.io/offensiveProgrammingBook_v1.2.2/semantic-names.html). If all you'd like to do is to use variable names like `a1, a2, a3, a11, a22, a33` for things that have actualy meanings such as distances, angles, density, etc., then I'd say there's no chance for the bot to help you write docstring.

This also applies to any other language where you would like to write comments. It won't give you perfect comment lines but it can certainly help. Occasionally, it can be distracting, especially what you are trying to do is so sophisticated and there's no way that the bot can figure out anything based on the variable names, function names, and the context. You might think if it's something that simple even a bot can figure out, there is probably no need to write much comments if any at all. Well, ask yourself to explain that 'simple' code three months later and see how much you still remember.

#### Comments driven code suggestions

As mentioned above, GitHub CoPilot can be used to write comment. At the same time, it can also write codes based on comments. So, you can write some comments (with the help of the bot), then you can wait for the bot to suggest codes for you. I've already showed that the bot is capable of getting simple things done as discussed in the [rotation matrix example](#automatically-generating-routines-slightly-different-from-previous-ones). Once you start using it, you will be surprised by the bot's capabilities in understanding what you want to do. Sometimes, it even figures things out before you have a clear idea what you should really do. I give you an example which probably cannot be reproduced considering the randomness of the process.

```python
def add_rectangle_manual(gmsh: gmsh, corner, dx, dy):
    """
    Manually add a rectangle to the model using the occ factory by adding points
    lines, and surfaces. The rectangle is defined by the coordinates of the
    lower left corner of the rectangle, the length of the rectangle in the x-
    and y-directions. Returns the tags of the added points, lines, curvloops,
    and surface

    Parameters
    ----------
    gmsh: Gmsh object
        The Gmsh object
    corner: list of floats
        The coordinates of the lower left corner of the rectangle
    dx: float
        The length of the rectangle in the x-direction
    dy: float
        The length of the rectangle in the y-direction

    Returns
    -------
    ptags: list of int
        The tags of the four points of the rectangle
    ltags: list of int
        The tags of the four lines of the rectangle
    stag: list of int (only 1 element)
        The tag of the surface of the rectangle
    """

    # Define occ as the factory
    factory = gmsh.model.occ

    # Calculate the coordinates of the four vertices of the rectangle
    verts = get_rectangle_coords(corner, dx, dy)

    # Add the four vertices of the rectangle to the model
    ptags = []
    for idx in range(4):
        ptag = factory.addPoint(verts[idx, 0], verts[idx, 1], verts[idx, 2])
        ptags.append(ptag)

    # Add the four lines of the rectangle to the model
    ltags = []
    for i in range(4):
        p1 = ptags[i]
        p2 = ptags[(i + 1) % 4]
        ltag = factory.addLine(p1, p2)
        ltags.append(ltag)

    # Add the curveloop of the rectangle to the model
    ctag = factory.addCurveLoop(ltags)

    # Add the surface defined by the curveloop
    stag = factory.addPlaneSurface([ctag])

    # Synchornize the model
    factory.synchronize()

    return ptags, ltags, [stag]
```

The code tries to add a rectangle to the [gmsh object](https://gmsh.info/) by adding points, lines, curve loops, and plane surface. This function just performs one of the many steps required to generate a 3D mesh for my EM modeling code using the Gmsh library. What surprised me at the time when I was writing this was the bot seems to know each function pretty well and it automatically fills in the variables. I guess this is because it was trained on codes that contain those open-source libraries that actually use the Gmsh library. Also, it's clever enough to figure out that I need to use the `mod (%)` operator to get the correct index for `p2` in this line: `p2 = ptags[(i + 1) % 4]`. When it AI-completed the code, I did not even know (or could not remember) that `%` is just the `mod` operator in Python. Of course, I found out it was correct after searching on Google just to verify its answer. Another line that also surprised me was this line: `stag = factory.addPlaneSurface([ctag])`. It knows that the function takes in a list instead of an integer. I did not know although I could figure out using the hover thing to have a quick look at the function. My initial guess was the function must be accepting integers.

This example is particularly interesting in that the bot seems to have a much better understanding of the library than me at the time when I first started working on it. It actually saved me time to look up the definitions of the functions (later I just blindly accept its suggestions) most of the times. Unfortunately, when it comes to really sophisticated circumstances, it can mess up spectacularly. So, eventually, I still spend some good amount of time and become familiar with the Gmsh library. But I still believe it's made the process shorter and much smoother.

### How the ChatGPT extension can be useful

#### Explain selected codes

I was working with Peter's Fortran code used to figure out the indices of edges of each triangular cell in a given mesh. I came across the following do loop and my brain simply refuses to work as I have not had a break for a while. So, I asked ChatGPT to explain what is this!

```fortran
! Loop over each cell in ugrid2:
DO i=1,ncells2

    ! Get the nodes for the current cell in ugrid2:
!         IF (mapnodes) THEN
!            CALL ugrid_get_cell_nodes(ugrid2,i,nn,inodes2,error); CALL er1(error,e); IF (e) RETURN
!            ! Find the nodes in ugrid1 corresponding to those nodes from ugrid2:
!            inodes1(1:nn) = imap(inodes2(1:nn))
!         ELSE
    CALL ugrid_get_cell_nodes(ugrid2,i,nn,inodes1,error); CALL er1(error,e); IF (e) RETURN
!         END IF

    ! Check for any nodes that could not be mapped from ugrid2 to ugrid1:
!         IF (ANY(inodes1(1:nn)<=0)) PRINT *, 'NODE(S) COULD NOT BE MAPPED'
    IF (ANY(inodes1(1:nn)<=0)) CYCLE ! the current cell in ugrid2 can't possibly be in ugrid1

    ! Find cells in ugrid1 containing all the nodes for the current cell in ugrid2:
    CALL ugrid_find_cells_from_nodes(ugrid1,nn,inodes1,nc,icells,error,allflag_op=allflag); CALL er1(error,e); IF (e) RETURN
!         IF (nc<=0) PRINT *, 'CELL COULD NOT BE CROSS-MAPPED', nn, inodes1(1:nn)
    IF (nc<=0) CYCLE

    ! Loop over those cells found:
    DO j=1,nc

    k = icells(j) ! index of a cell in ugrid1
    n = cross_membership(k,1) + 1 ! counter for the number of ugrid2 cells assigned to the current ugrid1 cell
    cross_membership(k,1) = n
    n = n + 1 ! index in cross_membership array that needs to be altered

    ! Set face edge information:
    IF (n>SIZE(cross_membership,2)) THEN
!               CALL ugrid_cell_indices_to_att(ugrid1,error); CALL er1(error,e); IF (e) RETURN
!               CALL ugrid_write_vtu(ugrid1,'junk1.vtu',.FALSE.,error); CALL er1(error,e); IF (e) RETURN
!               CALL ugrid_cell_indices_to_att(ugrid2,error); CALL er1(error,e); IF (e) RETURN
!               CALL ugrid_write_vtu(ugrid2,'junk2.vtu',.FALSE.,error); CALL er1(error,e); IF (e) RETURN
        CALL error_debug(error,'ugrid_other_sub','ugrid_calc_cross_membership','not enough space in output array', &
                        (/n,SIZE(cross_membership,2),k,i,cross_membership(k,:)/))
        RETURN
    END IF
    cross_membership(k,n) = i ! i is an index of a ugrid2 cell

    END DO

END DO
```

Here is the answer:

> This code is looping over each cell in the second grid (ugrid2). It then gets the list of nodes associated with each cell. If the mapping flag is set, it finds the nodes in the first grid (ugrid1) corresponding to the nodes from the second grid. Then it checks if any of the nodes could not be mapped from ugrid2 to ugrid1 and if so, skips the cell. Else, it finds all cells in the first grid containing all the nodes for the current cell in the second grid and assigns the cell in ugrid2 to that cell in ugrid1 by adding it to the output array (cross membership). Finally, the face edge information is set in case there is not enough space in the output array.

I will leave this to Peter to comment on whether the bot got it even close. To me, it's pretty close! So, imagine if you are like me and you are reading someone else's code of which you have no idea. Would you want to ask ChatGPT to explain it for you to get a rough idea? Although it might be wrong, you have to figure it out eventually by yourself anyways!

### Optimize selected code

Again, for the same code, I asked ChatGPT to optimize. Here's what I got (reformatted for a better display):

```fortran
! You can optimize the code by removing unused variables, consolidating 
! multiple "IF" statements into one, and using more descriptive variable 
! names. Below is an example of how to do this:
! Loop over each cell in ugrid2:
DO cell_idx = 1,ncells2

  ! Get the nodes for the current cell in ugrid2:
  CALL ugrid_get_cell_nodes(ugrid2,cell_idx,num_nodes,node_idxs2,error); CALL er1(error,e); IF (e) RETURN
  
  ! Find the nodes in ugrid1 corresponding to those nodes from ugrid2:
  node_idxs1(1:num_nodes) = imap(node_idxs2(1:num_nodes))
  
  ! Check for any nodes that could not be mapped from ugrid2 to ugrid1:
  IF (ANY(node_idxs1(1:num_nodes)<=0)) CYCLE ! the current cell in ugrid2 can't possibly be in ugrid1
  
  ! Find cells in ugrid1 containing all the nodes for the current cell in ugrid2:
  CALL ugrid_find_cells_from_nodes(ugrid1,num_nodes,node_idxs1,num_cells,corr_cell_idxs,&
                                   error,allflag_op=allflag)
  CALL er1(error,e); IF (e) RETURN
  IF (num_cells<=0) CYCLE

  ! Loop over those cells found:
  DO corr_cell_idx = 1,num_cells

    cross_map_idx = corr_cell_idxs(corr_cell_idx) ! index of a cell in ugrid1
    num_cells_mapped = cross_membership(cross_map_idx,1) + 1 ! counter for the number of ugrid2 cells assigned to the current ugrid1 cell
    cross_membership(cross_map_idx,1) = num_cells_mapped
    cross_idx_to_update = num_cells_mapped + 1 ! index in cross_membership array that needs to be altered
 
    ! Set face edge information:
    IF (cross_idx_to_update>SIZE(cross_membership,2)) THEN
      CALL error_debug(error,'ugrid_other_sub','ugrid_calc_cross_membership',&
                      'not enough space in output array', &
                      (/num_nodes,SIZE(cross_membership,2),cross_map_idx,cell_idx,cross_membership(cross_map_idx,:)/))
      RETURN
    END IF
    cross_membership(cross_map_idx,cross_idx_to_update) = cell_idx ! cell_idx is an index of a ugrid2 cell

  END DO

END DO
```

I did not check the correctness of the code. The bot is obviously not happy with how the variables are named and I do think after renaming them, the code becomes a bit easier to understand.

#### Find bugs in selected code

Lot's of times it's just give me suggestions as the code I have tested indeed does not have bugs. You can try it with a buggy code and see what happens.

#### Add comments for selected codes

#### Add tests for selected codes

I still do not understand much about how testing should be done in general!

#### Complet code

Didn't try.

#### Ad-hoc prompt

As a non-native English speaker, I would select a paragraph and ask it to check grammar and rephrase!