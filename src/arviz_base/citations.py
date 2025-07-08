"""How to cite ArviZ and its methods."""

import os
import re


def citations(methods=None, filepath=None, format_type="bibtex"):
    """
    List citations for ArviZ and the methods implemented in ArviZ.

    Parameters
    ----------
    methods : List
        Methods implemented in ArviZ from which to retrieve citations.
    filepath : str, optional
        Specifies the location to save the file with the citations.
        If ``None``, the result is returned as a string.
    format_type : str
       Specifies in which format the references will be displayed.
       Currently, only "bibtex" is supported.

    Examples
    --------
    >>> from arviz_base import citations
    >>> from arviz_stats import rhat
    >>> citations(methods=[rhat])  # Returns how to cite rhat
    >>> citations()  # Returns how to cite ArviZ
    """
    if methods is None:
        keys = {"Abril-Pla202X"}
    else:
        keys = set()
        for method in methods:
            matches = set(re.findall(r":(?:cite|footcite):[tp]:`(.*?)`", method.__doc__))
            keys.update(matches)
    ref_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "references.bib")
    with open(ref_path, encoding="utf-8") as fr:
        content = fr.read()

    if format_type == "bibtex":
        cite_text = _citation_bibtex(content, keys)
        if filepath:
            with open(filepath, "w") as fw:
                fw.write(cite_text)
        else:
            return cite_text
    else:
        raise ValueError("Invalid value for format_type. Use 'bibtex'.")


def _citation_bibtex(content, keys):
    """Extract and return references in BibTeX format."""
    extracted_refs = []
    for key in keys:
        match = re.search(rf"(@\w+\{{\s*{key}\s*,.*?\n\}})", content, re.DOTALL)
        extracted_refs.append(match.group(1))
    return "\n".join(extracted_refs)
