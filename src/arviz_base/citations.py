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
    >>> citations(methods=[rhat])  # Returns how to cite ArviZ and rhat
    >>> citations()  # Returns how to cite ArviZ
    """
    method_citations = [{"doi": "10.21105/joss.XXXXX"}]
    if methods is not None:
        for method in methods:
            _extract_ids_per_entry(method_citations, method.__doc__)

    if format_type == "bibtex":
        header = _get_header(methods)
        citation_text = _find_bibtex_entries(header, method_citations)
        if filepath:
            with open(filepath, "w") as fw:
                fw.write(citation_text)
        else:
            return citation_text
    else:
        raise ValueError("Invalid value for format_type. Use 'bibtex'.")


def _extract_ids_per_entry(data, text):
    entries = re.split(r"\n\s*\.\. \[\d+\] ", text.strip())

    doi_pattern = re.compile(r"https?://doi\.org/(\S+)", re.IGNORECASE)
    url_pattern = re.compile(r"https?://(?!doi\.org)(\S+)", re.IGNORECASE)

    for entry in entries:
        doi_match = doi_pattern.search(entry)
        if doi_match:
            doi = doi_match.group(1).rstrip(".")
            data.append({"doi": doi})
        else:
            urls = [url.rstrip(".") for url in url_pattern.findall(entry)]
            if urls:
                data.append({"urls": urls})
    return data


def _find_bibtex_entries(header, data):
    ref_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "references.bib")
    with open(ref_path, encoding="utf-8") as fr:
        bibtex_text = fr.read()

    entries = re.split(r"\n(?=@)", bibtex_text)

    doi_field_pattern = re.compile(r'doi\s*=\s*[{"]([^}"]+)[}"]', re.IGNORECASE)
    url_field_pattern = re.compile(r'url\s*=\s*[{"]([^}"]+)[}"]', re.IGNORECASE)

    references = [header]
    for identifier in data:
        found_entry = ""
        for entry in entries:
            bib_dois = doi_field_pattern.findall(entry)
            bib_urls = url_field_pattern.findall(entry)

            if "doi" in identifier and any(identifier["doi"] in doi for doi in bib_dois):
                found_entry = entry.strip()
                break

            if "urls" in identifier and any(
                any(url in b_url or b_url in url for b_url in bib_urls)
                for url in identifier["urls"]
            ):
                found_entry = entry.strip()
                break
        if found_entry:
            if found_entry not in references:
                references.append(found_entry)

    return "\n\n".join(references)


def _get_header(methods=None):
    references = "Bibtex format citations for ArviZ paper\n"

    if methods is not None:
        methods_str = ", ".join([method.__name__ for method in methods])
        references = references.strip() + f", and\nfor the following methods: {methods_str}"

    return references
