\documentclass[12pt, a4paper]{article}
\usepackage[margin=1in]{geometry}
\usepackage{enumitem}
\usepackage{hyperref}
\usepackage{setspace}
\setstretch{1.2}

% Formatting for section headers
\newcommand{\sectiontitle}[1]{\vspace{1em}\textbf{\Large #1}\vspace{0.5em}\\}

\begin{document}

% Header Section
\begin{center}
    {\LARGE \textbf{Name in English (Surname, Given Names)}} \\
    (Include all aliases and names in local language as well as English) \\
    \vspace{1em}
    \textbf{Home Address and Country} \\
    \textbf{Phone Numbers:} Primary, Secondary, Work, Home, Mobile \\
    \textbf{Email:} Provide all current email addresses (primary, secondary, work, personal, educational). \\
\end{center}

% Education Section
\sectiontitle{Education}
Please list ALL degrees attained or schools attended (since high school), including certificate programs and academic internships, beginning with the most recent. Include descriptions or topics of academic seminars. For any gaps, please explain what you were doing.
\begin{itemize}[left=0em, label={}]
    \item \textbf{Institution Name:} Example University (\url{www.example.edu}) \\
          \textbf{Address:} City, Country \\
          \textbf{Dates of Attendance:} MM/YYYY to MM/YYYY \\
          \textbf{Degree:} Bachelor of Science \\
          \textbf{Major:} Mechanical Engineering \\
          \textbf{Year Degree Received:} MM/YYYY \\
          \textbf{Thesis Topic/Research Focus:} Include 2-3 sentences describing your research topic. Consider including a list of courses or transcripts if applicable.
\end{itemize}

% Master's Thesis Example
\begin{itemize}[left=0em, label={}]
    \item \textbf{MM/YYYY to MM/YYYY:} Master’s Thesis Project \\
          \textbf{Institution Name:} General Hospital - City \\
          \begin{itemize}[left=1em]
              \item Creating a model for an adaptive prosthetic hand using EMG, vision, and language.
              \item Designing a prosthetic hand on XYZ software and 3-D printing.
              \item Building hardware for finger actuation.
              \item Analyzing grasp control for picking up objects.
          \end{itemize}
\end{itemize}

% Employment Section
\sectiontitle{Employment History}
List ALL work experience (full-time, part-time, internships, paid or unpaid). For any gaps, explain what you were doing.
\begin{itemize}[left=0em, label={}]
    \item \textbf{Employer Name:} Example Biotech Inc. \\
          \textbf{Address:} City, Country \\
          \textbf{Dates of Employment:} MM/YYYY to MM/YYYY \\
          \textbf{Job Title:} Microbiologist \\
          \textbf{Job Description:}
          \begin{itemize}[left=1em]
              \item Collect and analyze biological data about relationships between organisms and the environment.
              \item Completed melt curve analysis for genotype and bacteria identification.
              \item Used laboratory equipment to examine characteristics and classify microorganisms in water.
          \end{itemize}
\end{itemize}

% Awards and Memberships Section
\sectiontitle{Awards and Group Memberships}
\begin{itemize}[left=0em, label={}]
    \item Award Name (Year): Description of the award.
    \item Membership: Example Group or Society.
\end{itemize}

% Publications Section
\sectiontitle{Publications}
\begin{itemize}[left=0em, label={}]
    \item \textbf{Conference Presentations:}
          \begin{itemize}[left=1em]
              \item Smith, J. (2022). “Emerging Trends in Artificial Intelligence.” Keynote Speech at the International Conference on Technology Innovation, New York, NY.
          \end{itemize}
    \item \textbf{Written Publications:}
          \begin{itemize}[left=1em]
              \item Johnson, A., \& Smith, J. (2023). “E-commerce in plastics manufacturing.” \textit{American Plastics Foundation Monthly}, 312: 111-117.
          \end{itemize}
\end{itemize}

% Travel History Section
\sectiontitle{Travel History (Last 5 Years)}
List all countries visited in the past five years with the month and year.
\begin{itemize}[left=0em, label={}]
    \item South Korea (June 2019)
    \item United States (July 2020)
    \item Malaysia (August 2021)
\end{itemize}

\end{document}

