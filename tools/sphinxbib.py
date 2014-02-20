from __future__ import print_function
import os
import argparse
from bibpy import bib


def format_author(authors):
    items = []
    for author in authors:
        if 'given' not in author:
            author_split = author['family'].split()
            family = author_split[-1]
            given_split = author_split[:-1]
        else:
            given_split = author['given'].split(' ')
            family = author['family']

        given_initials = given_split[0][0] + '.'
        if len(given_split) > 1:
            given_initials += ' ' + (' '.join(given_split[1:]))

        items.append('%s, %s' % (family, given_initials))
    return '; '.join(items)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('bibtex')
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()
    with open(args.bibtex) as f:
        lines = os.linesep.join(l.strip() for l in f)
    bibobject = bib.Bibparser(lines, verbose=args.verbose)
    bibobject.parse()
    entries = sorted(bibobject.records.values(), key=lambda e: e['issued']['literal'])

    print('Publications')
    print('============\n')
    try:
        for entry in entries:
            url = entry['URL'] if 'URL' in entry else entry['url']
            print('`%s <%s>`_' % (entry['title'], url))
            print('~'*80)
            print()
            print(format_author(entry['author']), '*%s* **%s**, %s %s' %
                  (entry['journal'], entry['issued']['literal'],
                   entry['volume'], entry['page']))
            print()
            print('    ', entry['abstract'])   
            print()
    except KeyError as e:
        raise KeyError('Entry failure during processeing of %s' % entry)

if __name__ == '__main__':
    main()
