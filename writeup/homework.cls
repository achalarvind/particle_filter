\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{homework}[2015/02/15]

\LoadClass[11pt]{article}

\RequirePackage[dvipsnames]{xcolor}
\RequirePackage[utf8]{inputenc} % set input encoding (not needed with XeLaTeX)
%%% PAGE DIMENSIONS
\RequirePackage[a4paper,margin=0.5in]{geometry} % to change the page dimensions
\RequirePackage{graphicx} % support the \includegraphics command and options
% \RequirePackage[parfill]{parskip} % Activate to begin paragraphs with an empty line rather than an indent
%%% PACKAGES
\RequirePackage{booktabs} % for much better looking tables
\RequirePackage{array} % for better arrays (eg matrices) in maths
\RequirePackage{paralist} % very flexible & customisable lists (eg. enumerate/itemize, etc.)
\RequirePackage{verbatim} % adds environment for commenting out blocks of text & for better verbatim
\RequirePackage{subfig} % make it possible to include more than one captioned figure/table in a single float
\RequirePackage{placeins}
%%% HEADERS & FOOTERS
\RequirePackage{fancyhdr} % This should be set AFTER setting up the page geometry
\pagestyle{fancy} % options: empty , plain , fancy
\renewcommand{\headrulewidth}{0pt} % customise the layout...
\lhead{}\chead{}\rhead{}
\lfoot{}\cfoot{\thepage}\rfoot{}
%%% section* TITLE APPEARANCE
\RequirePackage{sectsty}
\allsectionsfont{\fontsize{12}{15}\sffamily\mdseries\upshape} % (See the fntguide.pdf for font help)
\sectionfont{\fontsize{14}{15}\sffamily\mdseries\upshape}
\subsubsectionfont{\fontsize{12}{15}\sffamily\mdseries\upshape} 
%%% ToC (table of contents) APPEARANCE
\RequirePackage[nottoc,notlof,notlot]{tocbibind} % Put the bibliography in the ToC
\RequirePackage[titles,subfigure]{tocloft} % Alter the style of the Table of Contents
\renewcommand{\cftsecfont}{\rmfamily\mdseries\upshape}
\renewcommand{\cftsecpagefont}{\rmfamily\mdseries\upshape} % No bold!
\RequirePackage{hyperref}
\RequirePackage{amsmath}
\RequirePackage{amssymb}
\RequirePackage{listings}
\RequirePackage[makeroom]{cancel}
\hypersetup{
    colorlinks,
    linkcolor={red!50!black},
    citecolor={blue!50!black},
    urlcolor={blue!80!black}
}
\RequirePackage{relsize}
\RequirePackage{titling}
\setlength{\droptitle}{-5em}

\RequirePackage{leftidx}

%%% UNCOMMENT TO USE MINTED, THEN EDIT YOUR LATEX COMPILER SETTING TO RUN WITH ''-shell-escape''
%\RequirePackage{minted}
%\renewcommand{\theFancyVerbLine}{
%  \sffamily\textcolor[rgb]{0.5,0.5,0.5}{\scriptsize\arabic{FancyVerbLine}}}
%  

%%%% TO USE MINTED:
%\begin{minted}[mathescape,
%               frame=lines,
%               framesep=2mm]{python}
% YOUR CODE HERE
%\end{minted}


%%% END Article customizations