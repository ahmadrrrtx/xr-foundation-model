"""Text wrapping and filling.
"""

# Copyright (C) 1999-2001 Gregory P. Ward.
# Copyright (C) 2002, 2003 Python Software Foundation.
# Written by Greg Ward <gward@python.net>

import re

__all__ = ['TextWrapper', 'wrap', 'fill', 'dedent', 'indent', 'shorten']

# Hardcode the recognized whitespace characters to the US-ASCII
# whitespace characters.  The main reason for doing this is that
# some Unicode spaces (like \u00a0) are non-breaking whitespaces.
_whitespace = '\t\n\x0b\x0c\r '

class TextWrapper:
    """
    Object for wrapping/filling text.  The public interface consists of
    the wrap() and fill() methods; the other methods are just there for
    subclasses to override in order to tweak the default behaviour.
    If you want to completely replace the main wrapping algorithm,
    you'll probably have to override _wrap_chunks().

    Several instance attributes control various aspects of wrapping:
      width (default: 70)
        the maximum width of wrapped lines (unless break_long_words
        is false)
      initial_indent (default: "")
        string that will be prepended to the first line of wrapped
        output.  Counts towards the line's width.
      subsequent_indent (default: "")
        string that will be prepended to all lines save the first
        of wrapped output; also counts towards each line's width.
      expand_tabs (default: true)
        Expand tabs in input text to spaces before further processing.
        Each tab will become 0 .. 'tabsize' spaces, depending on its position
        in its line.  If false, each tab is treated as a single character.
      tabsize (default: 8)
        Expand tabs in input text to 0 .. 'tabsize' spaces, unless
        'expand_tabs' is false.
      replace_whitespace (default: true)
        Replace all whitespace characters in the input text by spaces
        after tab expansion.  Note that if expand_tabs is false and
        replace_whitespace is true, every tab will be converted to a
        single space!
      fix_sentence_endings (default: false)
        Ensure that sentence-ending punctuation is always followed
        by two spaces.  Off by default because the algorithm is
        (unavoidably) imperfect.
      break_long_words (default: true)
        Break words longer than 'width'.  If false, those words will not
        be broken, and some lines might be longer than 'width'.
      break_on_hyphens (default: true)
        Allow breaking hyphenated words. If true, wrapping will occur
        preferably on whitespaces and right after hyphens part of
        compound words.
      drop_whitespace (default: true)
        Drop leading and trailing whitespace from lines.
      max_lines (default: None)
        Truncate wrapped lines.
      placeholder (default: ' [...]')
        Append to the last line of truncated text.
    """

    unicode_whitespace_trans = dict.fromkeys(map(ord, _whitespace), ord(' '))

    # This funky little regex is just the trick for splitting
    # text up into word-wrappable chunks.  E.g.
    #   "Hello there -- you goof-ball, use the -b option!"
    # splits into
    #   Hello/ /there/ /--/ /you/ /goof-/ball,/ /use/ /the/ /-b/ /option!
    # (after stripping out empty strings).
    word_punct = r'[\w!"\'&.,?]'
    letter = r'[^\d\W]'
    whitespace = r'[%s]' % re.escape(_whitespace)
    nowhitespace = '[^' + whitespace[1:]
    wordsep_re = re.compile(r'''
        ( # any whitespace
          %(ws)s+
        | # em-dash between words
          (?<=%(wp)s) -{2,} (?=\w)
        | # word, possibly hyphenated
          %(nws)s+? (?:
            # hyphenated word
              -(?: (?<=%(lt)s{2}-) | (?<=%(lt)s-%(lt)s-))
              (?= %(lt)s -? %(lt)s)
            | # end of word
              (?=%(ws)s|\Z)
            | # em-dash
              (?<=%(wp)s) (?=-{2,}\w)
            )
        )''' % {'wp': word_punct, 'lt': letter,
                'ws': whitespace, 'nws': nowhitespace},
        re.VERBOSE)
    del word_punct, letter, nowhitespace

    # This less funky little regex just split on recognized spaces. E.g.
    #   "Hello there -- you goof-ball, use the -b option!"
    # splits into
    #   Hello/ /there/ /--/ /you/ /goof-ball,/ /use/ /the/ /-b/ /option!/
    wordsep_simple_re = re.compile(r'(%s+)' % whitespace)
    del whitespace

    # XXX this is not locale- or charset-aware -- string.lowercase
    # is US-ASCII only (and therefore English-only)
    sentence_end_re = re.compile(r'[a-z]'             # lowercase letter
                                 r'[\.\!\?]'          # sentence-ending punct.
                                 r'[\"\']?'           # optional end-of-quote
                                 r'\Z')               # end of chunk

    def __init__(self,
                 width=70,
                 initial_indent="",
                 subsequent_indent="",
                 expand_tabs=True,
                 replace_whitespace=True,
                 fix_sentence_endings=False,
                 break_long_words=True,
                 drop_whitespace=True,
                 break_on_hyphens=True,
                 tabsize=8,
                 *,
                 max_lines=None,
                 placeholder=' [...]'):
        self.width = width
        self.initial_indent = initial_indent
        self.subsequent_indent = subsequent_indent
        self.expand_tabs = expand_tabs
        self.replace_whitespace = replace_whitespace
        self.fix_sentence_endings = fix_sentence_endings
        self.break_long_words = break_long_words
        self.drop_whitespace = drop_whitespace
        self.break_on_hyphens = break_on_hyphens
        self.tabsize = tabsize
        self.max_lines = max_lines
        self.placeholder = placeholder


    # -- Private methods -----------------------------------------------
    # (possibly useful for subclasses to override)

    def _munge_whitespace(self, text):
        """_munge_whitespace(text : string) -> string

        Munge whitespace in text: expand tabs and convert all other
        whitespace characters to spaces.  Eg. " foo\\tbar\\n\\nbaz"
        becomes " foo    bar  baz".
        """
        if self.expand_tabs:
            text = text.expandtabs(self.tabsize)
        if self.replace_whitespace:
            text = text.translate(self.unicode_whitespace_trans)
        return text


    def _split(self, text):
        """_split(text : string) -> [string]

        Split the text to wrap into indivisible chunks.  Chunks are
        not quite the same as words; see _wrap_chunks() for full
        details.  As an example, the text
          Look, goof-ball -- use the -b option!
        breaks into the following chunks:
          'Look,', ' ', 'goof-', 'ball', ' ', '--', ' ',
          'use', ' ', 'the', ' ', '-b', ' ', 'option!'
        if break_on_hyphens is True, or in:
          'Look,', ' ', 'goof-ball', ' ', '--', ' ',
          'use', ' ', 'the', ' ', '-b', ' ', option!'
        otherwise.
        """
        if self.break_on_hyphens is True:
            chunks = self.wordsep_re.split(text)
        else:
            chunks = self.wordsep_simple_re.split(text)
        chunks = [c for c in chunks if c]
        return chunks

    def _fix_sentence_endings(self, chunks):
        """_fix_sentence_endings(chunks : [string])

        Correct for sentence endings buried in 'chunks'.  Eg. when the
        original text contains "... foo.\\nBar ...", munge_whitespace()
        and split() will convert that to [..., "foo.", " ", "Bar", ...]
        which has one too few spaces; this method simply changes the one
        space to two.
        """
        i = 0
        patsearch = self.sentence_end_re.search
        while i < len(chunks)-1:
            if chunks[i+1] == " " and patsearch(chunks[i]):
                chunks[i+1] = "  "
                i += 2
            else:
                i += 1

    def _handle_long_word(self, reversed_chunks, cur_line, cur_len, width):
        """_handle_long_word(chunks : [string],
                             cur_line : [string],
                             cur_len : int, width : int)

        Handle a chunk of text (most likely a word, not whitespace) that
        is too long to fit in any line.
        """
        # Figure out when indent is larger than the specified width, and make
        # sure at least one character is stripped off on every pass
        if width < 1:
            space_left = 1
        else:
            space_left = width - cur_len

        # If we're allowed to break long words, then do so: put as much
        # of the next chunk onto the current line as will fit.
        if self.break_long_words and space_left > 0:
            end = space_left
            chunk = reversed_chunks[-1]
            if self.break_on_hyphens and len(chunk) > space_left:
                # break after last hyphen, but only if there are
                # non-hyphens before it
                hyphen = chunk.rfind('-', 0, space_left)
                if hyphen > 0 and any(c != '-' for c in chunk[:hyphen]):
                    end = hyphen + 1
            cur_line.append(chunk[:end])
            reversed_chunks[-1] = chunk[end:]

        # Otherwise, we have to preserve the long word intact.  Only add
        # it to the current line if there's nothing already there --
        # that minimizes how much we violate the width constraint.
        elif not cur_line:
            cur_line.append(reversed_chunks.pop())

        # If we're not allowed to break long words, and there's already
        # text on the current line, do nothing.  Next time through the
        # main loop of _wrap_chunks(), we'll wind up here again, but
        # cur_len will be zero, so the next line will be entirely
        # devoted to the long word that we can't handle right now.

    def _wrap_chunks(self, chunks):
        """_wrap_chunks(chunks : [string]) -> [string]

        Wrap a sequence of text chunks and return a list of lines of
        length 'self.width' or less.  (If 'break_long_words' is false,
        some lines may be longer than this.)  Chunks correspond roughly
        to words and the whitespace between them: each chunk is
        indivisible (modulo 'break_long_words'), but a line break can
        come between any two chunks.  Chunks should not have internal
        whitespace; ie. a chunk is either all whitespace or a "word".
        Whitespace chunks will be removed from the beginning and end of
        lines, but apart from that whitespace is preserved.
        """
        lines = []
        if self.width <= 0:
            raise ValueError("invalid width %r (must be > 0)" % self.width)
        if self.max_lines is not None:
            if self.max_lines > 1:
                indent = self.subsequent_indent
            else:
                indent = self.initial_indent
            if len(indent) + len(self.placeholder.lstrip()) > self.width:
                raise ValueError("placeholder too large for max width")

        # Arrange in reverse order so items can be efficiently popped
        # from a stack of chucks.
        chunks.reverse()

        while chunks:

            # Start the list of chunks that will make up the current line.
            # cur_len is just the length of all the chunks in cur_line.
            cur_line = []
            cur_len = 0

            # Figure out which static string will prefix this line.
            if lines:
                indent = self.subsequent_indent
            else:
                indent = self.initial_indent

            # Maximum width for this line.
            width = self.width - len(indent)

            # First chunk on line is whitespace -- drop it, unless this
            # is the very beginning of the text (ie. no lines started yet).
            if self.drop_whitespace and chunks[-1].strip() == '' and lines:
                del chunks[-1]

            while chunks:
                l = len(chunks[-1])

                # Can at least squeeze this chunk onto the current line.
                if cur_len + l <= width:
                    cur_line.append(chunks.pop())
                    cur_len += l

                # Nope, this line is full.
                else:
                    break

            # The current line is full, and the next chunk is too big to
            # fit on *any* line (not just this one).
            if chunks and len(chunks[-1]) > width:
                self._handle_long_word(chunks, cur_line, cur_len, width)
                cur_len = sum(map(len, cur_line))

            # If the last chunk on this line is all whitespace, drop it.
            if self.drop_whitespace and cur_line and cur_line[-1].strip() == '':
                cur_len -= len(cur_line[-1])
                del cur_line[-1]

            if cur_line:
                if (self.max_lines is None or
                    len(lines) + 1 < self.max_lines or
                    (not chunks or
                     self.drop_whitespace and
                     len(chunks) == 1 and
                     not chunks[0].strip()) and cur_len <= width):
                    # Convert current line back to a string and store it in
                    # list of all lines (return value).
                    lines.append(indent + ''.join(cur_line))
                else:
                    while cur_line:
                        if (cur_line[-1].strip() and
                            cur_len + len(self.placeholder) <= width):
                            cur_line.append(self.placeholder)
                            lines.append(indent + ''.join(cur_line))
                            break
                        cur_len -= len(cur_line[-1])
                        del cur_line[-1]
                    else:
                        if lines:
                            prev_line = lines[-1].rstrip()
                            if (len(prev_line) + len(self.placeholder) <=
                                    self.width):
                                lines[-1] = prev_line + self.placeholder
                                break
                        lines.append(indent + self.placeholder.lstrip())
                    break

        return lines

    def _split_chunks(self, text):
        text = self._munge_whitespace(text)
        return self._split(text)

    # -- Public interface ----------------------------------------------

    def wrap(self, text):
        """wrap(text : string) -> [string]

        Reformat the single paragraph in 'text' so it fits in lines of
        no more than 'self.width' columns, and return a list of wrapped
        lines.  Tabs in 'text' are expanded with string.expandtabs(),
        and all other whitespace characters (including newline) are
        converted to space.
        """
        chunks = self._split_chunks(text)
        if self.fix_sentence_endings:
            self._fix_sentence_endings(chunks)
        return self._wrap_chunks(chunks)

    def fill(self, text):
        """fill(text : string) -> string

        Reformat the single paragraph in 'text' to fit in lines of no
        more than 'self.width' columns, and return a new string
        containing the entire wrapped paragraph.
        """
        return "\n".join(self.wrap(text))


# -- Convenience interface ---------------------------------------------

def wrap(text, width=70, **kwargs):
    """Wrap a single paragraph of text, returning a list of wrapped lines.

    Reformat the single paragraph in 'text' so it fits in lines of no
    more than 'width' columns, and return a list of wrapped lines.  By
    default, tabs in 'text' are expanded with string.expandtabs(), and
    all other whitespace characters (including newline) are converted to
    space.  See TextWrapper class for available keyword args to customize
    wrapping behaviour.
    """
    w = TextWrapper(width=width, **kwargs)
    return w.wrap(text)

def fill(text, width=70, **kwargs):
    """Fill a single paragraph of text, returning a new string.

    Reformat the single paragraph in 'text' to fit in lines of no more
    than 'width' columns, and return a new string containing the entire
    wrapped paragraph.  As with wrap(), tabs are expanded and other
    whitespace characters converted to space.  See TextWrapper class for
    available keyword args to customize wrapping behaviour.
    """
    w = TextWrapper(width=width, **kwargs)
    return w.fill(text)

def shorten(text, width, **kwargs):
    """Collapse and truncate the given text to fit in the given width.

    The text first has its whitespace collapsed.  If it then fits in
    the *width*, it is returned as is.  Otherwise, as many words
    as possible are joined and then the placeholder is appended::

        >>> textwrap.shorten("Hello  world!", width=12)
        'Hello world!'
        >>> textwrap.shorten("Hello  world!", width=11)
        'Hello [...]'
    """
    w = TextWrapper(width=width, max_lines=1, **kwargs)
    return w.fill(' '.join(text.strip().split()))


# -- Loosely related functionality -------------------------------------

_whitespace_only_re = re.compile('^[ \t]+$', re.MULTILINE)
_leading_whitespace_re = re.compile('(^[ \t]*)(?:[^ \t\n])', re.MULTILINE)

def dedent(text):
    """Remove any common leading whitespace from every line in `text`.

    This can be used to make triple-quoted strings line up with the left
    edge of the display, while still presenting them in the source code
    in indented form.

    Note that tabs and spaces are both treated as whitespace, but they
    are not equal: the lines "  hello" and "\\thello" are
    considered to have no common leading whitespace.

    Entirely blank lines are normalized to a newline character.
    """
    # Look for the longest leading string of spaces and tabs common to
    # all lines.
    margin = None
    text = _whitespace_only_re.sub('', text)
    indents = _leading_whitespace_re.findall(text)
    for indent in indents:
        if margin is None:
            margin = indent

        # Current line more deeply indented than previous winner:
        # no change (previous winner is still on top).
        elif indent.startswith(margin):
            pass

        # Current line consistent with and no deeper than previous winner:
        # it's the new winner.
        elif margin.startswith(indent):
            margin = indent

        # Find the largest common whitespace between current line and previous
        # winner.
        else:
            for i, (x, y) in enumerate(zip(margin, indent)):
                if x != y:
                    margin = margin[:i]
                    break

    # sanity check (testing/debugging only)
    if 0 and margin:
        for line in text.split("\n"):
            assert not line or line.startswith(margin), \
                   "line = %r, margin = %r" % (line, margin)

    if margin:
        text = re.sub(r'(?m)^' + margin, '', text)
    return text


def indent(text, prefix, predicate=None):
    """Adds 'prefix' to the beginning of selected lines in 'text'.

    If 'predicate' is provided, 'prefix' will only be added to the lines
    where 'predicate(line)' is True. If 'predicate' is not provided,
    it will default to adding 'prefix' to all non-empty lines that do not
    consist solely of whitespace characters.
    """
    if predicate is None:
        # str.splitlines(True) doesn't produce empty string.
        #  ''.splitlines(True) => []
        #  'foo\n'.splitlines(True) => ['foo\n']
        # So we can use just `not s.isspace()` here.
        predicate = lambda s: not s.isspace()

    prefixed_lines = []
    for line in text.splitlines(True):
        if predicate(line):
            prefixed_lines.append(prefix)
        prefixed_lines.append(line)

    return ''.join(prefixed_lines)


if __name__ == "__main__":
    #print dedent("\tfoo\n\tbar")
    #print dedent("  \thello there\n  \t  how are you?")
    print(dedent("Hello there.\n  This is indented."))



"""Bisection algorithms."""


def insort_right(a, x, lo=0, hi=None, *, key=None):
    """Insert item x in list a, and keep it sorted assuming a is sorted.

    If x is already in a, insert it to the right of the rightmost x.

    Optional args lo (default 0) and hi (default len(a)) bound the
    slice of a to be searched.

    A custom key function can be supplied to customize the sort order.
    """
    if key is None:
        lo = bisect_right(a, x, lo, hi)
    else:
        lo = bisect_right(a, key(x), lo, hi, key=key)
    a.insert(lo, x)


def bisect_right(a, x, lo=0, hi=None, *, key=None):
    """Return the index where to insert item x in list a, assuming a is sorted.

    The return value i is such that all e in a[:i] have e <= x, and all e in
    a[i:] have e > x.  So if x already appears in the list, a.insert(i, x) will
    insert just after the rightmost x already there.

    Optional args lo (default 0) and hi (default len(a)) bound the
    slice of a to be searched.

    A custom key function can be supplied to customize the sort order.
    """

    if lo < 0:
        raise ValueError('lo must be non-negative')
    if hi is None:
        hi = len(a)
    # Note, the comparison uses "<" to match the
    # __lt__() logic in list.sort() and in heapq.
    if key is None:
        while lo < hi:
            mid = (lo + hi) // 2
            if x < a[mid]:
                hi = mid
            else:
                lo = mid + 1
    else:
        while lo < hi:
            mid = (lo + hi) // 2
            if x < key(a[mid]):
                hi = mid
            else:
                lo = mid + 1
    return lo


def insort_left(a, x, lo=0, hi=None, *, key=None):
    """Insert item x in list a, and keep it sorted assuming a is sorted.

    If x is already in a, insert it to the left of the leftmost x.

    Optional args lo (default 0) and hi (default len(a)) bound the
    slice of a to be searched.

    A custom key function can be supplied to customize the sort order.
    """

    if key is None:
        lo = bisect_left(a, x, lo, hi)
    else:
        lo = bisect_left(a, key(x), lo, hi, key=key)
    a.insert(lo, x)

def bisect_left(a, x, lo=0, hi=None, *, key=None):
    """Return the index where to insert item x in list a, assuming a is sorted.

    The return value i is such that all e in a[:i] have e < x, and all e in
    a[i:] have e >= x.  So if x already appears in the list, a.insert(i, x) will
    insert just before the leftmost x already there.

    Optional args lo (default 0) and hi (default len(a)) bound the
    slice of a to be searched.

    A custom key function can be supplied to customize the sort order.
    """

    if lo < 0:
        raise ValueError('lo must be non-negative')
    if hi is None:
        hi = len(a)
    # Note, the comparison uses "<" to match the
    # __lt__() logic in list.sort() and in heapq.
    if key is None:
        while lo < hi:
            mid = (lo + hi) // 2
            if a[mid] < x:
                lo = mid + 1
            else:
                hi = mid
    else:
        while lo < hi:
            mid = (lo + hi) // 2
            if key(a[mid]) < x:
                lo = mid + 1
            else:
                hi = mid
    return lo


# Overwrite above definitions with a fast C implementation
try:
    from _bisect import *
except ImportError:
    pass

# Create aliases
bisect = bisect_right
insort = insort_right



"""Heap queue algorithm (a.k.a. priority queue).

Heaps are arrays for which a[k] <= a[2*k+1] and a[k] <= a[2*k+2] for
all k, counting elements from 0.  For the sake of comparison,
non-existing elements are considered to be infinite.  The interesting
property of a heap is that a[0] is always its smallest element.

Usage:

heap = []            # creates an empty heap
heappush(heap, item) # pushes a new item on the heap
item = heappop(heap) # pops the smallest item from the heap
item = heap[0]       # smallest item on the heap without popping it
heapify(x)           # transforms list into a heap, in-place, in linear time
item = heappushpop(heap, item) # pushes a new item and then returns
                               # the smallest item; the heap size is unchanged
item = heapreplace(heap, item) # pops and returns smallest item, and adds
                               # new item; the heap size is unchanged

Our API differs from textbook heap algorithms as follows:

- We use 0-based indexing.  This makes the relationship between the
  index for a node and the indexes for its children slightly less
  obvious, but is more suitable since Python uses 0-based indexing.

- Our heappop() method returns the smallest item, not the largest.

These two make it possible to view the heap as a regular Python list
without surprises: heap[0] is the smallest item, and heap.sort()
maintains the heap invariant!
"""

# Original code by Kevin O'Connor, augmented by Tim Peters and Raymond Hettinger

__about__ = """Heap queues

[explanation by François Pinard]

Heaps are arrays for which a[k] <= a[2*k+1] and a[k] <= a[2*k+2] for
all k, counting elements from 0.  For the sake of comparison,
non-existing elements are considered to be infinite.  The interesting
property of a heap is that a[0] is always its smallest element.

The strange invariant above is meant to be an efficient memory
representation for a tournament.  The numbers below are `k', not a[k]:

                                   0

                  1                                 2

          3               4                5               6

      7       8       9       10      11      12      13      14

    15 16   17 18   19 20   21 22   23 24   25 26   27 28   29 30


In the tree above, each cell `k' is topping `2*k+1' and `2*k+2'.  In
a usual binary tournament we see in sports, each cell is the winner
over the two cells it tops, and we can trace the winner down the tree
to see all opponents s/he had.  However, in many computer applications
of such tournaments, we do not need to trace the history of a winner.
To be more memory efficient, when a winner is promoted, we try to
replace it by something else at a lower level, and the rule becomes
that a cell and the two cells it tops contain three different items,
but the top cell "wins" over the two topped cells.

If this heap invariant is protected at all time, index 0 is clearly
the overall winner.  The simplest algorithmic way to remove it and
find the "next" winner is to move some loser (let's say cell 30 in the
diagram above) into the 0 position, and then percolate this new 0 down
the tree, exchanging values, until the invariant is re-established.
This is clearly logarithmic on the total number of items in the tree.
By iterating over all items, you get an O(n ln n) sort.

A nice feature of this sort is that you can efficiently insert new
items while the sort is going on, provided that the inserted items are
not "better" than the last 0'th element you extracted.  This is
especially useful in simulation contexts, where the tree holds all
incoming events, and the "win" condition means the smallest scheduled
time.  When an event schedule other events for execution, they are
scheduled into the future, so they can easily go into the heap.  So, a
heap is a good structure for implementing schedulers (this is what I
used for my MIDI sequencer :-).

Various structures for implementing schedulers have been extensively
studied, and heaps are good for this, as they are reasonably speedy,
the speed is almost constant, and the worst case is not much different
than the average case.  However, there are other representations which
are more efficient overall, yet the worst cases might be terrible.

Heaps are also very useful in big disk sorts.  You most probably all
know that a big sort implies producing "runs" (which are pre-sorted
sequences, which size is usually related to the amount of CPU memory),
followed by a merging passes for these runs, which merging is often
very cleverly organised[1].  It is very important that the initial
sort produces the longest runs possible.  Tournaments are a good way
to that.  If, using all the memory available to hold a tournament, you
replace and percolate items that happen to fit the current run, you'll
produce runs which are twice the size of the memory for random input,
and much better for input fuzzily ordered.

Moreover, if you output the 0'th item on disk and get an input which
may not fit in the current tournament (because the value "wins" over
the last output value), it cannot fit in the heap, so the size of the
heap decreases.  The freed memory could be cleverly reused immediately
for progressively building a second heap, which grows at exactly the
same rate the first heap is melting.  When the first heap completely
vanishes, you switch heaps and start a new run.  Clever and quite
effective!

In a word, heaps are useful memory structures to know.  I use them in
a few applications, and I think it is good to keep a `heap' module
around. :-)

--------------------
[1] The disk balancing algorithms which are current, nowadays, are
more annoying than clever, and this is a consequence of the seeking
capabilities of the disks.  On devices which cannot seek, like big
tape drives, the story was quite different, and one had to be very
clever to ensure (far in advance) that each tape movement will be the
most effective possible (that is, will best participate at
"progressing" the merge).  Some tapes were even able to read
backwards, and this was also used to avoid the rewinding time.
Believe me, real good tape sorts were quite spectacular to watch!
From all times, sorting has always been a Great Art! :-)
"""

__all__ = ['heappush', 'heappop', 'heapify', 'heapreplace', 'merge',
           'nlargest', 'nsmallest', 'heappushpop']

def heappush(heap, item):
    """Push item onto heap, maintaining the heap invariant."""
    heap.append(item)
    _siftdown(heap, 0, len(heap)-1)

def heappop(heap):
    """Pop the smallest item off the heap, maintaining the heap invariant."""
    lastelt = heap.pop()    # raises appropriate IndexError if heap is empty
    if heap:
        returnitem = heap[0]
        heap[0] = lastelt
        _siftup(heap, 0)
        return returnitem
    return lastelt

def heapreplace(heap, item):
    """Pop and return the current smallest value, and add the new item.

    This is more efficient than heappop() followed by heappush(), and can be
    more appropriate when using a fixed-size heap.  Note that the value
    returned may be larger than item!  That constrains reasonable uses of
    this routine unless written as part of a conditional replacement:

        if item > heap[0]:
            item = heapreplace(heap, item)
    """
    returnitem = heap[0]    # raises appropriate IndexError if heap is empty
    heap[0] = item
    _siftup(heap, 0)
    return returnitem

def heappushpop(heap, item):
    """Fast version of a heappush followed by a heappop."""
    if heap and heap[0] < item:
        item, heap[0] = heap[0], item
        _siftup(heap, 0)
    return item

def heapify(x):
    """Transform list into a heap, in-place, in O(len(x)) time."""
    n = len(x)
    # Transform bottom-up.  The largest index there's any point to looking at
    # is the largest with a child index in-range, so must have 2*i + 1 < n,
    # or i < (n-1)/2.  If n is even = 2*j, this is (2*j-1)/2 = j-1/2 so
    # j-1 is the largest, which is n//2 - 1.  If n is odd = 2*j+1, this is
    # (2*j+1-1)/2 = j so j-1 is the largest, and that's again n//2-1.
    for i in reversed(range(n//2)):
        _siftup(x, i)

def _heappop_max(heap):
    """Maxheap version of a heappop."""
    lastelt = heap.pop()    # raises appropriate IndexError if heap is empty
    if heap:
        returnitem = heap[0]
        heap[0] = lastelt
        _siftup_max(heap, 0)
        return returnitem
    return lastelt

def _heapreplace_max(heap, item):
    """Maxheap version of a heappop followed by a heappush."""
    returnitem = heap[0]    # raises appropriate IndexError if heap is empty
    heap[0] = item
    _siftup_max(heap, 0)
    return returnitem

def _heapify_max(x):
    """Transform list into a maxheap, in-place, in O(len(x)) time."""
    n = len(x)
    for i in reversed(range(n//2)):
        _siftup_max(x, i)

# 'heap' is a heap at all indices >= startpos, except possibly for pos.  pos
# is the index of a leaf with a possibly out-of-order value.  Restore the
# heap invariant.
def _siftdown(heap, startpos, pos):
    newitem = heap[pos]
    # Follow the path to the root, moving parents down until finding a place
    # newitem fits.
    while pos > startpos:
        parentpos = (pos - 1) >> 1
        parent = heap[parentpos]
        if newitem < parent:
            heap[pos] = parent
            pos = parentpos
            continue
        break
    heap[pos] = newitem

# The child indices of heap index pos are already heaps, and we want to make
# a heap at index pos too.  We do this by bubbling the smaller child of
# pos up (and so on with that child's children, etc) until hitting a leaf,
# then using _siftdown to move the oddball originally at index pos into place.
#
# We *could* break out of the loop as soon as we find a pos where newitem <=
# both its children, but turns out that's not a good idea, and despite that
# many books write the algorithm that way.  During a heap pop, the last array
# element is sifted in, and that tends to be large, so that comparing it
# against values starting from the root usually doesn't pay (= usually doesn't
# get us out of the loop early).  See Knuth, Volume 3, where this is
# explained and quantified in an exercise.
#
# Cutting the # of comparisons is important, since these routines have no
# way to extract "the priority" from an array element, so that intelligence
# is likely to be hiding in custom comparison methods, or in array elements
# storing (priority, record) tuples.  Comparisons are thus potentially
# expensive.
#
# On random arrays of length 1000, making this change cut the number of
# comparisons made by heapify() a little, and those made by exhaustive
# heappop() a lot, in accord with theory.  Here are typical results from 3
# runs (3 just to demonstrate how small the variance is):
#
# Compares needed by heapify     Compares needed by 1000 heappops
# --------------------------     --------------------------------
# 1837 cut to 1663               14996 cut to 8680
# 1855 cut to 1659               14966 cut to 8678
# 1847 cut to 1660               15024 cut to 8703
#
# Building the heap by using heappush() 1000 times instead required
# 2198, 2148, and 2219 compares:  heapify() is more efficient, when
# you can use it.
#
# The total compares needed by list.sort() on the same lists were 8627,
# 8627, and 8632 (this should be compared to the sum of heapify() and
# heappop() compares):  list.sort() is (unsurprisingly!) more efficient
# for sorting.

def _siftup(heap, pos):
    endpos = len(heap)
    startpos = pos
    newitem = heap[pos]
    # Bubble up the smaller child until hitting a leaf.
    childpos = 2*pos + 1    # leftmost child position
    while childpos < endpos:
        # Set childpos to index of smaller child.
        rightpos = childpos + 1
        if rightpos < endpos and not heap[childpos] < heap[rightpos]:
            childpos = rightpos
        # Move the smaller child up.
        heap[pos] = heap[childpos]
        pos = childpos
        childpos = 2*pos + 1
    # The leaf at pos is empty now.  Put newitem there, and bubble it up
    # to its final resting place (by sifting its parents down).
    heap[pos] = newitem
    _siftdown(heap, startpos, pos)

def _siftdown_max(heap, startpos, pos):
    'Maxheap variant of _siftdown'
    newitem = heap[pos]
    # Follow the path to the root, moving parents down until finding a place
    # newitem fits.
    while pos > startpos:
        parentpos = (pos - 1) >> 1
        parent = heap[parentpos]
        if parent < newitem:
            heap[pos] = parent
            pos = parentpos
            continue
        break
    heap[pos] = newitem

def _siftup_max(heap, pos):
    'Maxheap variant of _siftup'
    endpos = len(heap)
    startpos = pos
    newitem = heap[pos]
    # Bubble up the larger child until hitting a leaf.
    childpos = 2*pos + 1    # leftmost child position
    while childpos < endpos:
        # Set childpos to index of larger child.
        rightpos = childpos + 1
        if rightpos < endpos and not heap[rightpos] < heap[childpos]:
            childpos = rightpos
        # Move the larger child up.
        heap[pos] = heap[childpos]
        pos = childpos
        childpos = 2*pos + 1
    # The leaf at pos is empty now.  Put newitem there, and bubble it up
    # to its final resting place (by sifting its parents down).
    heap[pos] = newitem
    _siftdown_max(heap, startpos, pos)

def merge(*iterables, key=None, reverse=False):
    '''Merge multiple sorted inputs into a single sorted output.

    Similar to sorted(itertools.chain(*iterables)) but returns a generator,
    does not pull the data into memory all at once, and assumes that each of
    the input streams is already sorted (smallest to largest).

    >>> list(merge([1,3,5,7], [0,2,4,8], [5,10,15,20], [], [25]))
    [0, 1, 2, 3, 4, 5, 5, 7, 8, 10, 15, 20, 25]

    If *key* is not None, applies a key function to each element to determine
    its sort order.

    >>> list(merge(['dog', 'horse'], ['cat', 'fish', 'kangaroo'], key=len))
    ['dog', 'cat', 'fish', 'horse', 'kangaroo']

    '''

    h = []
    h_append = h.append

    if reverse:
        _heapify = _heapify_max
        _heappop = _heappop_max
        _heapreplace = _heapreplace_max
        direction = -1
    else:
        _heapify = heapify
        _heappop = heappop
        _heapreplace = heapreplace
        direction = 1

    if key is None:
        for order, it in enumerate(map(iter, iterables)):
            try:
                next = it.__next__
                h_append([next(), order * direction, next])
            except StopIteration:
                pass
        _heapify(h)
        while len(h) > 1:
            try:
                while True:
                    value, order, next = s = h[0]
                    yield value
                    s[0] = next()           # raises StopIteration when exhausted
                    _heapreplace(h, s)      # restore heap condition
            except StopIteration:
                _heappop(h)                 # remove empty iterator
        if h:
            # fast case when only a single iterator remains
            value, order, next = h[0]
            yield value
            yield from next.__self__
        return

    for order, it in enumerate(map(iter, iterables)):
        try:
            next = it.__next__
            value = next()
            h_append([key(value), order * direction, value, next])
        except StopIteration:
            pass
    _heapify(h)
    while len(h) > 1:
        try:
            while True:
                key_value, order, value, next = s = h[0]
                yield value
                value = next()
                s[0] = key(value)
                s[2] = value
                _heapreplace(h, s)
        except StopIteration:
            _heappop(h)
    if h:
        key_value, order, value, next = h[0]
        yield value
        yield from next.__self__


# Algorithm notes for nlargest() and nsmallest()
# ==============================================
#
# Make a single pass over the data while keeping the k most extreme values
# in a heap.  Memory consumption is limited to keeping k values in a list.
#
# Measured performance for random inputs:
#
#                                   number of comparisons
#    n inputs     k-extreme values  (average of 5 trials)   % more than min()
# -------------   ----------------  ---------------------   -----------------
#      1,000           100                  3,317               231.7%
#     10,000           100                 14,046                40.5%
#    100,000           100                105,749                 5.7%
#  1,000,000           100              1,007,751                 0.8%
# 10,000,000           100             10,009,401                 0.1%
#
# Theoretical number of comparisons for k smallest of n random inputs:
#
# Step   Comparisons                  Action
# ----   --------------------------   ---------------------------
#  1     1.66 * k                     heapify the first k-inputs
#  2     n - k                        compare remaining elements to top of heap
#  3     k * (1 + lg2(k)) * ln(n/k)   replace the topmost value on the heap
#  4     k * lg2(k) - (k/2)           final sort of the k most extreme values
#
# Combining and simplifying for a rough estimate gives:
#
#        comparisons = n + k * (log(k, 2) * log(n/k) + log(k, 2) + log(n/k))
#
# Computing the number of comparisons for step 3:
# -----------------------------------------------
# * For the i-th new value from the iterable, the probability of being in the
#   k most extreme values is k/i.  For example, the probability of the 101st
#   value seen being in the 100 most extreme values is 100/101.
# * If the value is a new extreme value, the cost of inserting it into the
#   heap is 1 + log(k, 2).
# * The probability times the cost gives:
#            (k/i) * (1 + log(k, 2))
# * Summing across the remaining n-k elements gives:
#            sum((k/i) * (1 + log(k, 2)) for i in range(k+1, n+1))
# * This reduces to:
#            (H(n) - H(k)) * k * (1 + log(k, 2))
# * Where H(n) is the n-th harmonic number estimated by:
#            gamma = 0.5772156649
#            H(n) = log(n, e) + gamma + 1 / (2 * n)
#   http://en.wikipedia.org/wiki/Harmonic_series_(mathematics)#Rate_of_divergence
# * Substituting the H(n) formula:
#            comparisons = k * (1 + log(k, 2)) * (log(n/k, e) + (1/n - 1/k) / 2)
#
# Worst-case for step 3:
# ----------------------
# In the worst case, the input data is reversed sorted so that every new element
# must be inserted in the heap:
#
#             comparisons = 1.66 * k + log(k, 2) * (n - k)
#
# Alternative Algorithms
# ----------------------
# Other algorithms were not used because they:
# 1) Took much more auxiliary memory,
# 2) Made multiple passes over the data.
# 3) Made more comparisons in common cases (small k, large n, semi-random input).
# See the more detailed comparison of approach at:
# http://code.activestate.com/recipes/577573-compare-algorithms-for-heapqsmallest

def nsmallest(n, iterable, key=None):
    """Find the n smallest elements in a dataset.

    Equivalent to:  sorted(iterable, key=key)[:n]
    """

    # Short-cut for n==1 is to use min()
    if n == 1:
        it = iter(iterable)
        sentinel = object()
        result = min(it, default=sentinel, key=key)
        return [] if result is sentinel else [result]

    # When n>=size, it's faster to use sorted()
    try:
        size = len(iterable)
    except (TypeError, AttributeError):
        pass
    else:
        if n >= size:
            return sorted(iterable, key=key)[:n]

    # When key is none, use simpler decoration
    if key is None:
        it = iter(iterable)
        # put the range(n) first so that zip() doesn't
        # consume one too many elements from the iterator
        result = [(elem, i) for i, elem in zip(range(n), it)]
        if not result:
            return result
        _heapify_max(result)
        top = result[0][0]
        order = n
        _heapreplace = _heapreplace_max
        for elem in it:
            if elem < top:
                _heapreplace(result, (elem, order))
                top, _order = result[0]
                order += 1
        result.sort()
        return [elem for (elem, order) in result]

    # General case, slowest method
    it = iter(iterable)
    result = [(key(elem), i, elem) for i, elem in zip(range(n), it)]
    if not result:
        return result
    _heapify_max(result)
    top = result[0][0]
    order = n
    _heapreplace = _heapreplace_max
    for elem in it:
        k = key(elem)
        if k < top:
            _heapreplace(result, (k, order, elem))
            top, _order, _elem = result[0]
            order += 1
    result.sort()
    return [elem for (k, order, elem) in result]

def nlargest(n, iterable, key=None):
    """Find the n largest elements in a dataset.

    Equivalent to:  sorted(iterable, key=key, reverse=True)[:n]
    """

    # Short-cut for n==1 is to use max()
    if n == 1:
        it = iter(iterable)
        sentinel = object()
        result = max(it, default=sentinel, key=key)
        return [] if result is sentinel else [result]

    # When n>=size, it's faster to use sorted()
    try:
        size = len(iterable)
    except (TypeError, AttributeError):
        pass
    else:
        if n >= size:
            return sorted(iterable, key=key, reverse=True)[:n]

    # When key is none, use simpler decoration
    if key is None:
        it = iter(iterable)
        result = [(elem, i) for i, elem in zip(range(0, -n, -1), it)]
        if not result:
            return result
        heapify(result)
        top = result[0][0]
        order = -n
        _heapreplace = heapreplace
        for elem in it:
            if top < elem:
                _heapreplace(result, (elem, order))
                top, _order = result[0]
                order -= 1
        result.sort(reverse=True)
        return [elem for (elem, order) in result]

    # General case, slowest method
    it = iter(iterable)
    result = [(key(elem), i, elem) for i, elem in zip(range(0, -n, -1), it)]
    if not result:
        return result
    heapify(result)
    top = result[0][0]
    order = -n
    _heapreplace = heapreplace
    for elem in it:
        k = key(elem)
        if top < k:
            _heapreplace(result, (k, order, elem))
            top, _order, _elem = result[0]
            order -= 1
    result.sort(reverse=True)
    return [elem for (k, order, elem) in result]

# If available, use C implementation
try:
    from _heapq import *
except ImportError:
    pass
try:
    from _heapq import _heapreplace_max
except ImportError:
    pass
try:
    from _heapq import _heapify_max
except ImportError:
    pass
try:
    from _heapq import _heappop_max
except ImportError:
    pass


if __name__ == "__main__":

    import doctest # pragma: no cover
    print(doctest.testmod()) # pragma: no cover



"""
Basic statistics module.

This module provides functions for calculating statistics of data, including
averages, variance, and standard deviation.

Calculating averages
--------------------

==================  ==================================================
Function            Description
==================  ==================================================
mean                Arithmetic mean (average) of data.
fmean               Fast, floating-point arithmetic mean.
geometric_mean      Geometric mean of data.
harmonic_mean       Harmonic mean of data.
median              Median (middle value) of data.
median_low          Low median of data.
median_high         High median of data.
median_grouped      Median, or 50th percentile, of grouped data.
mode                Mode (most common value) of data.
multimode           List of modes (most common values of data).
quantiles           Divide data into intervals with equal probability.
==================  ==================================================

Calculate the arithmetic mean ("the average") of data:

>>> mean([-1.0, 2.5, 3.25, 5.75])
2.625


Calculate the standard median of discrete data:

>>> median([2, 3, 4, 5])
3.5


Calculate the median, or 50th percentile, of data grouped into class intervals
centred on the data values provided. E.g. if your data points are rounded to
the nearest whole number:

>>> median_grouped([2, 2, 3, 3, 3, 4])  #doctest: +ELLIPSIS
2.8333333333...

This should be interpreted in this way: you have two data points in the class
interval 1.5-2.5, three data points in the class interval 2.5-3.5, and one in
the class interval 3.5-4.5. The median of these data points is 2.8333...


Calculating variability or spread
---------------------------------

==================  =============================================
Function            Description
==================  =============================================
pvariance           Population variance of data.
variance            Sample variance of data.
pstdev              Population standard deviation of data.
stdev               Sample standard deviation of data.
==================  =============================================

Calculate the standard deviation of sample data:

>>> stdev([2.5, 3.25, 5.5, 11.25, 11.75])  #doctest: +ELLIPSIS
4.38961843444...

If you have previously calculated the mean, you can pass it as the optional
second argument to the four "spread" functions to avoid recalculating it:

>>> data = [1, 2, 2, 4, 4, 4, 5, 6]
>>> mu = mean(data)
>>> pvariance(data, mu)
2.5


Statistics for relations between two inputs
-------------------------------------------

==================  ====================================================
Function            Description
==================  ====================================================
covariance          Sample covariance for two variables.
correlation         Pearson's correlation coefficient for two variables.
linear_regression   Intercept and slope for simple linear regression.
==================  ====================================================

Calculate covariance, Pearson's correlation, and simple linear regression
for two inputs:

>>> x = [1, 2, 3, 4, 5, 6, 7, 8, 9]
>>> y = [1, 2, 3, 1, 2, 3, 1, 2, 3]
>>> covariance(x, y)
0.75
>>> correlation(x, y)  #doctest: +ELLIPSIS
0.31622776601...
>>> linear_regression(x, y)  #doctest:
LinearRegression(slope=0.1, intercept=1.5)


Exceptions
----------

A single exception is defined: StatisticsError is a subclass of ValueError.

"""

__all__ = [
    'NormalDist',
    'StatisticsError',
    'correlation',
    'covariance',
    'fmean',
    'geometric_mean',
    'harmonic_mean',
    'kde',
    'kde_random',
    'linear_regression',
    'mean',
    'median',
    'median_grouped',
    'median_high',
    'median_low',
    'mode',
    'multimode',
    'pstdev',
    'pvariance',
    'quantiles',
    'stdev',
    'variance',
]

import math
import numbers
import random
import sys

from fractions import Fraction
from decimal import Decimal
from itertools import count, groupby, repeat
from bisect import bisect_left, bisect_right
from math import hypot, sqrt, fabs, exp, erf, tau, log, fsum, sumprod
from math import isfinite, isinf, pi, cos, sin, tan, cosh, asin, atan, acos
from functools import reduce
from operator import itemgetter
from collections import Counter, namedtuple, defaultdict

_SQRT2 = sqrt(2.0)
_random = random

# === Exceptions ===

class StatisticsError(ValueError):
    pass


# === Private utilities ===

def _sum(data):
    """_sum(data) -> (type, sum, count)

    Return a high-precision sum of the given numeric data as a fraction,
    together with the type to be converted to and the count of items.

    Examples
    --------

    >>> _sum([3, 2.25, 4.5, -0.5, 0.25])
    (<class 'float'>, Fraction(19, 2), 5)

    Some sources of round-off error will be avoided:

    # Built-in sum returns zero.
    >>> _sum([1e50, 1, -1e50] * 1000)
    (<class 'float'>, Fraction(1000, 1), 3000)

    Fractions and Decimals are also supported:

    >>> from fractions import Fraction as F
    >>> _sum([F(2, 3), F(7, 5), F(1, 4), F(5, 6)])
    (<class 'fractions.Fraction'>, Fraction(63, 20), 4)

    >>> from decimal import Decimal as D
    >>> data = [D("0.1375"), D("0.2108"), D("0.3061"), D("0.0419")]
    >>> _sum(data)
    (<class 'decimal.Decimal'>, Fraction(6963, 10000), 4)

    Mixed types are currently treated as an error, except that int is
    allowed.
    """
    count = 0
    types = set()
    types_add = types.add
    partials = {}
    partials_get = partials.get
    for typ, values in groupby(data, type):
        types_add(typ)
        for n, d in map(_exact_ratio, values):
            count += 1
            partials[d] = partials_get(d, 0) + n
    if None in partials:
        # The sum will be a NAN or INF. We can ignore all the finite
        # partials, and just look at this special one.
        total = partials[None]
        assert not _isfinite(total)
    else:
        # Sum all the partial sums using builtin sum.
        total = sum(Fraction(n, d) for d, n in partials.items())
    T = reduce(_coerce, types, int)  # or raise TypeError
    return (T, total, count)


def _ss(data, c=None):
    """Return the exact mean and sum of square deviations of sequence data.

    Calculations are done in a single pass, allowing the input to be an iterator.

    If given *c* is used the mean; otherwise, it is calculated from the data.
    Use the *c* argument with care, as it can lead to garbage results.

    """
    if c is not None:
        T, ssd, count = _sum((d := x - c) * d for x in data)
        return (T, ssd, c, count)
    count = 0
    types = set()
    types_add = types.add
    sx_partials = defaultdict(int)
    sxx_partials = defaultdict(int)
    for typ, values in groupby(data, type):
        types_add(typ)
        for n, d in map(_exact_ratio, values):
            count += 1
            sx_partials[d] += n
            sxx_partials[d] += n * n
    if not count:
        ssd = c = Fraction(0)
    elif None in sx_partials:
        # The sum will be a NAN or INF. We can ignore all the finite
        # partials, and just look at this special one.
        ssd = c = sx_partials[None]
        assert not _isfinite(ssd)
    else:
        sx = sum(Fraction(n, d) for d, n in sx_partials.items())
        sxx = sum(Fraction(n, d*d) for d, n in sxx_partials.items())
        # This formula has poor numeric properties for floats,
        # but with fractions it is exact.
        ssd = (count * sxx - sx * sx) / count
        c = sx / count
    T = reduce(_coerce, types, int)  # or raise TypeError
    return (T, ssd, c, count)


def _isfinite(x):
    try:
        return x.is_finite()  # Likely a Decimal.
    except AttributeError:
        return math.isfinite(x)  # Coerces to float first.


def _coerce(T, S):
    """Coerce types T and S to a common type, or raise TypeError.

    Coercion rules are currently an implementation detail. See the CoerceTest
    test class in test_statistics for details.
    """
    # See http://bugs.python.org/issue24068.
    assert T is not bool, "initial type T is bool"
    # If the types are the same, no need to coerce anything. Put this
    # first, so that the usual case (no coercion needed) happens as soon
    # as possible.
    if T is S:  return T
    # Mixed int & other coerce to the other type.
    if S is int or S is bool:  return T
    if T is int:  return S
    # If one is a (strict) subclass of the other, coerce to the subclass.
    if issubclass(S, T):  return S
    if issubclass(T, S):  return T
    # Ints coerce to the other type.
    if issubclass(T, int):  return S
    if issubclass(S, int):  return T
    # Mixed fraction & float coerces to float (or float subclass).
    if issubclass(T, Fraction) and issubclass(S, float):
        return S
    if issubclass(T, float) and issubclass(S, Fraction):
        return T
    # Any other combination is disallowed.
    msg = "don't know how to coerce %s and %s"
    raise TypeError(msg % (T.__name__, S.__name__))


def _exact_ratio(x):
    """Return Real number x to exact (numerator, denominator) pair.

    >>> _exact_ratio(0.25)
    (1, 4)

    x is expected to be an int, Fraction, Decimal or float.
    """

    # XXX We should revisit whether using fractions to accumulate exact
    # ratios is the right way to go.

    # The integer ratios for binary floats can have numerators or
    # denominators with over 300 decimal digits.  The problem is more
    # acute with decimal floats where the default decimal context
    # supports a huge range of exponents from Emin=-999999 to
    # Emax=999999.  When expanded with as_integer_ratio(), numbers like
    # Decimal('3.14E+5000') and Decimal('3.14E-5000') have large
    # numerators or denominators that will slow computation.

    # When the integer ratios are accumulated as fractions, the size
    # grows to cover the full range from the smallest magnitude to the
    # largest.  For example, Fraction(3.14E+300) + Fraction(3.14E-300),
    # has a 616 digit numerator.  Likewise,
    # Fraction(Decimal('3.14E+5000')) + Fraction(Decimal('3.14E-5000'))
    # has 10,003 digit numerator.

    # This doesn't seem to have been problem in practice, but it is a
    # potential pitfall.

    try:
        return x.as_integer_ratio()
    except AttributeError:
        pass
    except (OverflowError, ValueError):
        # float NAN or INF.
        assert not _isfinite(x)
        return (x, None)
    try:
        # x may be an Integral ABC.
        return (x.numerator, x.denominator)
    except AttributeError:
        msg = f"can't convert type '{type(x).__name__}' to numerator/denominator"
        raise TypeError(msg)


def _convert(value, T):
    """Convert value to given numeric type T."""
    if type(value) is T:
        # This covers the cases where T is Fraction, or where value is
        # a NAN or INF (Decimal or float).
        return value
    if issubclass(T, int) and value.denominator != 1:
        T = float
    try:
        # FIXME: what do we do if this overflows?
        return T(value)
    except TypeError:
        if issubclass(T, Decimal):
            return T(value.numerator) / T(value.denominator)
        else:
            raise


def _fail_neg(values, errmsg='negative value'):
    """Iterate over values, failing if any are less than zero."""
    for x in values:
        if x < 0:
            raise StatisticsError(errmsg)
        yield x


def _rank(data, /, *, key=None, reverse=False, ties='average', start=1) -> list[float]:
    """Rank order a dataset. The lowest value has rank 1.

    Ties are averaged so that equal values receive the same rank:

        >>> data = [31, 56, 31, 25, 75, 18]
        >>> _rank(data)
        [3.5, 5.0, 3.5, 2.0, 6.0, 1.0]

    The operation is idempotent:

        >>> _rank([3.5, 5.0, 3.5, 2.0, 6.0, 1.0])
        [3.5, 5.0, 3.5, 2.0, 6.0, 1.0]

    It is possible to rank the data in reverse order so that the
    highest value has rank 1.  Also, a key-function can extract
    the field to be ranked:

        >>> goals = [('eagles', 45), ('bears', 48), ('lions', 44)]
        >>> _rank(goals, key=itemgetter(1), reverse=True)
        [2.0, 1.0, 3.0]

    Ranks are conventionally numbered starting from one; however,
    setting *start* to zero allows the ranks to be used as array indices:

        >>> prize = ['Gold', 'Silver', 'Bronze', 'Certificate']
        >>> scores = [8.1, 7.3, 9.4, 8.3]
        >>> [prize[int(i)] for i in _rank(scores, start=0, reverse=True)]
        ['Bronze', 'Certificate', 'Gold', 'Silver']

    """
    # If this function becomes public at some point, more thought
    # needs to be given to the signature.  A list of ints is
    # plausible when ties is "min" or "max".  When ties is "average",
    # either list[float] or list[Fraction] is plausible.

    # Default handling of ties matches scipy.stats.mstats.spearmanr.
    if ties != 'average':
        raise ValueError(f'Unknown tie resolution method: {ties!r}')
    if key is not None:
        data = map(key, data)
    val_pos = sorted(zip(data, count()), reverse=reverse)
    i = start - 1
    result = [0] * len(val_pos)
    for _, g in groupby(val_pos, key=itemgetter(0)):
        group = list(g)
        size = len(group)
        rank = i + (size + 1) / 2
        for value, orig_pos in group:
            result[orig_pos] = rank
        i += size
    return result


def _integer_sqrt_of_frac_rto(n: int, m: int) -> int:
    """Square root of n/m, rounded to the nearest integer using round-to-odd."""
    # Reference: https://www.lri.fr/~melquion/doc/05-imacs17_1-expose.pdf
    a = math.isqrt(n // m)
    return a | (a*a*m != n)


# For 53 bit precision floats, the bit width used in
# _float_sqrt_of_frac() is 109.
_sqrt_bit_width: int = 2 * sys.float_info.mant_dig + 3


def _float_sqrt_of_frac(n: int, m: int) -> float:
    """Square root of n/m as a float, correctly rounded."""
    # See principle and proof sketch at: https://bugs.python.org/msg407078
    q = (n.bit_length() - m.bit_length() - _sqrt_bit_width) // 2
    if q >= 0:
        numerator = _integer_sqrt_of_frac_rto(n, m << 2 * q) << q
        denominator = 1
    else:
        numerator = _integer_sqrt_of_frac_rto(n << -2 * q, m)
        denominator = 1 << -q
    return numerator / denominator   # Convert to float


def _decimal_sqrt_of_frac(n: int, m: int) -> Decimal:
    """Square root of n/m as a Decimal, correctly rounded."""
    # Premise:  For decimal, computing (n/m).sqrt() can be off
    #           by 1 ulp from the correctly rounded result.
    # Method:   Check the result, moving up or down a step if needed.
    if n <= 0:
        if not n:
            return Decimal('0.0')
        n, m = -n, -m

    root = (Decimal(n) / Decimal(m)).sqrt()
    nr, dr = root.as_integer_ratio()

    plus = root.next_plus()
    np, dp = plus.as_integer_ratio()
    # test: n / m > ((root + plus) / 2) ** 2
    if 4 * n * (dr*dp)**2 > m * (dr*np + dp*nr)**2:
        return plus

    minus = root.next_minus()
    nm, dm = minus.as_integer_ratio()
    # test: n / m < ((root + minus) / 2) ** 2
    if 4 * n * (dr*dm)**2 < m * (dr*nm + dm*nr)**2:
        return minus

    return root


# === Measures of central tendency (averages) ===

def mean(data):
    """Return the sample arithmetic mean of data.

    >>> mean([1, 2, 3, 4, 4])
    2.8

    >>> from fractions import Fraction as F
    >>> mean([F(3, 7), F(1, 21), F(5, 3), F(1, 3)])
    Fraction(13, 21)

    >>> from decimal import Decimal as D
    >>> mean([D("0.5"), D("0.75"), D("0.625"), D("0.375")])
    Decimal('0.5625')

    If ``data`` is empty, StatisticsError will be raised.
    """
    T, total, n = _sum(data)
    if n < 1:
        raise StatisticsError('mean requires at least one data point')
    return _convert(total / n, T)


def fmean(data, weights=None):
    """Convert data to floats and compute the arithmetic mean.

    This runs faster than the mean() function and it always returns a float.
    If the input dataset is empty, it raises a StatisticsError.

    >>> fmean([3.5, 4.0, 5.25])
    4.25
    """
    if weights is None:
        try:
            n = len(data)
        except TypeError:
            # Handle iterators that do not define __len__().
            n = 0
            def count(iterable):
                nonlocal n
                for n, x in enumerate(iterable, start=1):
                    yield x
            data = count(data)
        total = fsum(data)
        if not n:
            raise StatisticsError('fmean requires at least one data point')
        return total / n
    if not isinstance(weights, (list, tuple)):
        weights = list(weights)
    try:
        num = sumprod(data, weights)
    except ValueError:
        raise StatisticsError('data and weights must be the same length')
    den = fsum(weights)
    if not den:
        raise StatisticsError('sum of weights must be non-zero')
    return num / den


def geometric_mean(data):
    """Convert data to floats and compute the geometric mean.

    Raises a StatisticsError if the input dataset is empty
    or if it contains a negative value.

    Returns zero if the product of inputs is zero.

    No special efforts are made to achieve exact results.
    (However, this may change in the future.)

    >>> round(geometric_mean([54, 24, 36]), 9)
    36.0
    """
    n = 0
    found_zero = False
    def count_positive(iterable):
        nonlocal n, found_zero
        for n, x in enumerate(iterable, start=1):
            if x > 0.0 or math.isnan(x):
                yield x
            elif x == 0.0:
                found_zero = True
            else:
                raise StatisticsError('No negative inputs allowed', x)
    total = fsum(map(log, count_positive(data)))
    if not n:
        raise StatisticsError('Must have a non-empty dataset')
    if math.isnan(total):
        return math.nan
    if found_zero:
        return math.nan if total == math.inf else 0.0
    return exp(total / n)


def harmonic_mean(data, weights=None):
    """Return the harmonic mean of data.

    The harmonic mean is the reciprocal of the arithmetic mean of the
    reciprocals of the data.  It can be used for averaging ratios or
    rates, for example speeds.

    Suppose a car travels 40 km/hr for 5 km and then speeds-up to
    60 km/hr for another 5 km. What is the average speed?

        >>> harmonic_mean([40, 60])
        48.0

    Suppose a car travels 40 km/hr for 5 km, and when traffic clears,
    speeds-up to 60 km/hr for the remaining 30 km of the journey. What
    is the average speed?

        >>> harmonic_mean([40, 60], weights=[5, 30])
        56.0

    If ``data`` is empty, or any element is less than zero,
    ``harmonic_mean`` will raise ``StatisticsError``.
    """
    if iter(data) is data:
        data = list(data)
    errmsg = 'harmonic mean does not support negative values'
    n = len(data)
    if n < 1:
        raise StatisticsError('harmonic_mean requires at least one data point')
    elif n == 1 and weights is None:
        x = data[0]
        if isinstance(x, (numbers.Real, Decimal)):
            if x < 0:
                raise StatisticsError(errmsg)
            return x
        else:
            raise TypeError('unsupported type')
    if weights is None:
        weights = repeat(1, n)
        sum_weights = n
    else:
        if iter(weights) is weights:
            weights = list(weights)
        if len(weights) != n:
            raise StatisticsError('Number of weights does not match data size')
        _, sum_weights, _ = _sum(w for w in _fail_neg(weights, errmsg))
    try:
        data = _fail_neg(data, errmsg)
        T, total, count = _sum(w / x if w else 0 for w, x in zip(weights, data))
    except ZeroDivisionError:
        return 0
    if total <= 0:
        raise StatisticsError('Weighted sum must be positive')
    return _convert(sum_weights / total, T)

# FIXME: investigate ways to calculate medians without sorting? Quickselect?
def median(data):
    """Return the median (middle value) of numeric data.

    When the number of data points is odd, return the middle data point.
    When the number of data points is even, the median is interpolated by
    taking the average of the two middle values:

    >>> median([1, 3, 5])
    3
    >>> median([1, 3, 5, 7])
    4.0

    """
    data = sorted(data)
    n = len(data)
    if n == 0:
        raise StatisticsError("no median for empty data")
    if n % 2 == 1:
        return data[n // 2]
    else:
        i = n // 2
        return (data[i - 1] + data[i]) / 2


def median_low(data):
    """Return the low median of numeric data.

    When the number of data points is odd, the middle value is returned.
    When it is even, the smaller of the two middle values is returned.

    >>> median_low([1, 3, 5])
    3
    >>> median_low([1, 3, 5, 7])
    3

    """
    data = sorted(data)
    n = len(data)
    if n == 0:
        raise StatisticsError("no median for empty data")
    if n % 2 == 1:
        return data[n // 2]
    else:
        return data[n // 2 - 1]


def median_high(data):
    """Return the high median of data.

    When the number of data points is odd, the middle value is returned.
    When it is even, the larger of the two middle values is returned.

    >>> median_high([1, 3, 5])
    3
    >>> median_high([1, 3, 5, 7])
    5

    """
    data = sorted(data)
    n = len(data)
    if n == 0:
        raise StatisticsError("no median for empty data")
    return data[n // 2]


def median_grouped(data, interval=1.0):
    """Estimates the median for numeric data binned around the midpoints
    of consecutive, fixed-width intervals.

    The *data* can be any iterable of numeric data with each value being
    exactly the midpoint of a bin.  At least one value must be present.

    The *interval* is width of each bin.

    For example, demographic information may have been summarized into
    consecutive ten-year age groups with each group being represented
    by the 5-year midpoints of the intervals:

        >>> demographics = Counter({
        ...    25: 172,   # 20 to 30 years old
        ...    35: 484,   # 30 to 40 years old
        ...    45: 387,   # 40 to 50 years old
        ...    55:  22,   # 50 to 60 years old
        ...    65:   6,   # 60 to 70 years old
        ... })

    The 50th percentile (median) is the 536th person out of the 1071
    member cohort.  That person is in the 30 to 40 year old age group.

    The regular median() function would assume that everyone in the
    tricenarian age group was exactly 35 years old.  A more tenable
    assumption is that the 484 members of that age group are evenly
    distributed between 30 and 40.  For that, we use median_grouped().

        >>> data = list(demographics.elements())
        >>> median(data)
        35
        >>> round(median_grouped(data, interval=10), 1)
        37.5

    The caller is responsible for making sure the data points are separated
    by exact multiples of *interval*.  This is essential for getting a
    correct result.  The function does not check this precondition.

    Inputs may be any numeric type that can be coerced to a float during
    the interpolation step.

    """
    data = sorted(data)
    n = len(data)
    if not n:
        raise StatisticsError("no median for empty data")

    # Find the value at the midpoint. Remember this corresponds to the
    # midpoint of the class interval.
    x = data[n // 2]

    # Using O(log n) bisection, find where all the x values occur in the data.
    # All x will lie within data[i:j].
    i = bisect_left(data, x)
    j = bisect_right(data, x, lo=i)

    # Coerce to floats, raising a TypeError if not possible
    try:
        interval = float(interval)
        x = float(x)
    except ValueError:
        raise TypeError(f'Value cannot be converted to a float')

    # Interpolate the median using the formula found at:
    # https://www.cuemath.com/data/median-of-grouped-data/
    L = x - interval / 2.0    # Lower limit of the median interval
    cf = i                    # Cumulative frequency of the preceding interval
    f = j - i                 # Number of elements in the median internal
    return L + interval * (n / 2 - cf) / f


def mode(data):
    """Return the most common data point from discrete or nominal data.

    ``mode`` assumes discrete data, and returns a single value. This is the
    standard treatment of the mode as commonly taught in schools:

        >>> mode([1, 1, 2, 3, 3, 3, 3, 4])
        3

    This also works with nominal (non-numeric) data:

        >>> mode(["red", "blue", "blue", "red", "green", "red", "red"])
        'red'

    If there are multiple modes with same frequency, return the first one
    encountered:

        >>> mode(['red', 'red', 'green', 'blue', 'blue'])
        'red'

    If *data* is empty, ``mode``, raises StatisticsError.

    """
    pairs = Counter(iter(data)).most_common(1)
    try:
        return pairs[0][0]
    except IndexError:
        raise StatisticsError('no mode for empty data') from None


def multimode(data):
    """Return a list of the most frequently occurring values.

    Will return more than one result if there are multiple modes
    or an empty list if *data* is empty.

    >>> multimode('aabbbbbbbbcc')
    ['b']
    >>> multimode('aabbbbccddddeeffffgg')
    ['b', 'd', 'f']
    >>> multimode('')
    []
    """
    counts = Counter(iter(data))
    if not counts:
        return []
    maxcount = max(counts.values())
    return [value for value, count in counts.items() if count == maxcount]


def kde(data, h, kernel='normal', *, cumulative=False):
    """Kernel Density Estimation:  Create a continuous probability density
    function or cumulative distribution function from discrete samples.

    The basic idea is to smooth the data using a kernel function
    to help draw inferences about a population from a sample.

    The degree of smoothing is controlled by the scaling parameter h
    which is called the bandwidth.  Smaller values emphasize local
    features while larger values give smoother results.

    The kernel determines the relative weights of the sample data
    points.  Generally, the choice of kernel shape does not matter
    as much as the more influential bandwidth smoothing parameter.

    Kernels that give some weight to every sample point:

       normal (gauss)
       logistic
       sigmoid

    Kernels that only give weight to sample points within
    the bandwidth:

       rectangular (uniform)
       triangular
       parabolic (epanechnikov)
       quartic (biweight)
       triweight
       cosine

    If *cumulative* is true, will return a cumulative distribution function.

    A StatisticsError will be raised if the data sequence is empty.

    Example
    -------

    Given a sample of six data points, construct a continuous
    function that estimates the underlying probability density:

        >>> sample = [-2.1, -1.3, -0.4, 1.9, 5.1, 6.2]
        >>> f_hat = kde(sample, h=1.5)

    Compute the area under the curve:

        >>> area = sum(f_hat(x) for x in range(-20, 20))
        >>> round(area, 4)
        1.0

    Plot the estimated probability density function at
    evenly spaced points from -6 to 10:

        >>> for x in range(-6, 11):
        ...     density = f_hat(x)
        ...     plot = ' ' * int(density * 400) + 'x'
        ...     print(f'{x:2}: {density:.3f} {plot}')
        ...
        -6: 0.002 x
        -5: 0.009    x
        -4: 0.031             x
        -3: 0.070                             x
        -2: 0.111                                             x
        -1: 0.125                                                   x
         0: 0.110                                            x
         1: 0.086                                   x
         2: 0.068                            x
         3: 0.059                        x
         4: 0.066                           x
         5: 0.082                                 x
         6: 0.082                                 x
         7: 0.058                        x
         8: 0.028            x
         9: 0.009    x
        10: 0.002 x

    Estimate P(4.5 < X <= 7.5), the probability that a new sample value
    will be between 4.5 and 7.5:

        >>> cdf = kde(sample, h=1.5, cumulative=True)
        >>> round(cdf(7.5) - cdf(4.5), 2)
        0.22

    References
    ----------

    Kernel density estimation and its application:
    https://www.itm-conferences.org/articles/itmconf/pdf/2018/08/itmconf_sam2018_00037.pdf

    Kernel functions in common use:
    https://en.wikipedia.org/wiki/Kernel_(statistics)#kernel_functions_in_common_use

    Interactive graphical demonstration and exploration:
    https://demonstrations.wolfram.com/KernelDensityEstimation/

    Kernel estimation of cumulative distribution function of a random variable with bounded support
    https://www.econstor.eu/bitstream/10419/207829/1/10.21307_stattrans-2016-037.pdf

    """

    n = len(data)
    if not n:
        raise StatisticsError('Empty data sequence')

    if not isinstance(data[0], (int, float)):
        raise TypeError('Data sequence must contain ints or floats')

    if h <= 0.0:
        raise StatisticsError(f'Bandwidth h must be positive, not {h=!r}')

    match kernel:

        case 'normal' | 'gauss':
            sqrt2pi = sqrt(2 * pi)
            sqrt2 = sqrt(2)
            K = lambda t: exp(-1/2 * t * t) / sqrt2pi
            W = lambda t: 1/2 * (1.0 + erf(t / sqrt2))
            support = None

        case 'logistic':
            # 1.0 / (exp(t) + 2.0 + exp(-t))
            K = lambda t: 1/2 / (1.0 + cosh(t))
            W = lambda t: 1.0 - 1.0 / (exp(t) + 1.0)
            support = None

        case 'sigmoid':
            # (2/pi) / (exp(t) + exp(-t))
            c1 = 1 / pi
            c2 = 2 / pi
            K = lambda t: c1 / cosh(t)
            W = lambda t: c2 * atan(exp(t))
            support = None

        case 'rectangular' | 'uniform':
            K = lambda t: 1/2
            W = lambda t: 1/2 * t + 1/2
            support = 1.0

        case 'triangular':
            K = lambda t: 1.0 - abs(t)
            W = lambda t: t*t * (1/2 if t < 0.0 else -1/2) + t + 1/2
            support = 1.0

        case 'parabolic' | 'epanechnikov':
            K = lambda t: 3/4 * (1.0 - t * t)
            W = lambda t: -1/4 * t**3 + 3/4 * t + 1/2
            support = 1.0

        case 'quartic' | 'biweight':
            K = lambda t: 15/16 * (1.0 - t * t) ** 2
            W = lambda t: 3/16 * t**5 - 5/8 * t**3 + 15/16 * t + 1/2
            support = 1.0

        case 'triweight':
            K = lambda t: 35/32 * (1.0 - t * t) ** 3
            W = lambda t: 35/32 * (-1/7*t**7 + 3/5*t**5 - t**3 + t) + 1/2
            support = 1.0

        case 'cosine':
            c1 = pi / 4
            c2 = pi / 2
            K = lambda t: c1 * cos(c2 * t)
            W = lambda t: 1/2 * sin(c2 * t) + 1/2
            support = 1.0

        case _:
            raise StatisticsError(f'Unknown kernel name: {kernel!r}')

    if support is None:

        def pdf(x):
            n = len(data)
            return sum(K((x - x_i) / h) for x_i in data) / (n * h)

        def cdf(x):
            n = len(data)
            return sum(W((x - x_i) / h) for x_i in data) / n

    else:

        sample = sorted(data)
        bandwidth = h * support

        def pdf(x):
            nonlocal n, sample
            if len(data) != n:
                sample = sorted(data)
                n = len(data)
            i = bisect_left(sample, x - bandwidth)
            j = bisect_right(sample, x + bandwidth)
            supported = sample[i : j]
            return sum(K((x - x_i) / h) for x_i in supported) / (n * h)

        def cdf(x):
            nonlocal n, sample
            if len(data) != n:
                sample = sorted(data)
                n = len(data)
            i = bisect_left(sample, x - bandwidth)
            j = bisect_right(sample, x + bandwidth)
            supported = sample[i : j]
            return sum((W((x - x_i) / h) for x_i in supported), i) / n

    if cumulative:
        cdf.__doc__ = f'CDF estimate with {h=!r} and {kernel=!r}'
        return cdf

    else:
        pdf.__doc__ = f'PDF estimate with {h=!r} and {kernel=!r}'
        return pdf


# Notes on methods for computing quantiles
# ----------------------------------------
#
# There is no one perfect way to compute quantiles.  Here we offer
# two methods that serve common needs.  Most other packages
# surveyed offered at least one or both of these two, making them
# "standard" in the sense of "widely-adopted and reproducible".
# They are also easy to explain, easy to compute manually, and have
# straight-forward interpretations that aren't surprising.

# The default method is known as "R6", "PERCENTILE.EXC", or "expected
# value of rank order statistics". The alternative method is known as
# "R7", "PERCENTILE.INC", or "mode of rank order statistics".

# For sample data where there is a positive probability for values
# beyond the range of the data, the R6 exclusive method is a
# reasonable choice.  Consider a random sample of nine values from a
# population with a uniform distribution from 0.0 to 1.0.  The
# distribution of the third ranked sample point is described by
# betavariate(alpha=3, beta=7) which has mode=0.250, median=0.286, and
# mean=0.300.  Only the latter (which corresponds with R6) gives the
# desired cut point with 30% of the population falling below that
# value, making it comparable to a result from an inv_cdf() function.
# The R6 exclusive method is also idempotent.

# For describing population data where the end points are known to
# be included in the data, the R7 inclusive method is a reasonable
# choice.  Instead of the mean, it uses the mode of the beta
# distribution for the interior points.  Per Hyndman & Fan, "One nice
# property is that the vertices of Q7(p) divide the range into n - 1
# intervals, and exactly 100p% of the intervals lie to the left of
# Q7(p) and 100(1 - p)% of the intervals lie to the right of Q7(p)."

# If needed, other methods could be added.  However, for now, the
# position is that fewer options make for easier choices and that
# external packages can be used for anything more advanced.

def quantiles(data, *, n=4, method='exclusive'):
    """Divide *data* into *n* continuous intervals with equal probability.

    Returns a list of (n - 1) cut points separating the intervals.

    Set *n* to 4 for quartiles (the default).  Set *n* to 10 for deciles.
    Set *n* to 100 for percentiles which gives the 99 cuts points that
    separate *data* in to 100 equal sized groups.

    The *data* can be any iterable containing sample.
    The cut points are linearly interpolated between data points.

    If *method* is set to *inclusive*, *data* is treated as population
    data.  The minimum value is treated as the 0th percentile and the
    maximum value is treated as the 100th percentile.
    """
    if n < 1:
        raise StatisticsError('n must be at least 1')
    data = sorted(data)
    ld = len(data)
    if ld < 2:
        if ld == 1:
            return data * (n - 1)
        raise StatisticsError('must have at least one data point')

    if method == 'inclusive':
        m = ld - 1
        result = []
        for i in range(1, n):
            j, delta = divmod(i * m, n)
            interpolated = (data[j] * (n - delta) + data[j + 1] * delta) / n
            result.append(interpolated)
        return result

    if method == 'exclusive':
        m = ld + 1
        result = []
        for i in range(1, n):
            j = i * m // n                               # rescale i to m/n
            j = 1 if j < 1 else ld-1 if j > ld-1 else j  # clamp to 1 .. ld-1
            delta = i*m - j*n                            # exact integer math
            interpolated = (data[j - 1] * (n - delta) + data[j] * delta) / n
            result.append(interpolated)
        return result

    raise ValueError(f'Unknown method: {method!r}')


# === Measures of spread ===

# See http://mathworld.wolfram.com/Variance.html
#     http://mathworld.wolfram.com/SampleVariance.html


def variance(data, xbar=None):
    """Return the sample variance of data.

    data should be an iterable of Real-valued numbers, with at least two
    values. The optional argument xbar, if given, should be the mean of
    the data. If it is missing or None, the mean is automatically calculated.

    Use this function when your data is a sample from a population. To
    calculate the variance from the entire population, see ``pvariance``.

    Examples:

    >>> data = [2.75, 1.75, 1.25, 0.25, 0.5, 1.25, 3.5]
    >>> variance(data)
    1.3720238095238095

    If you have already calculated the mean of your data, you can pass it as
    the optional second argument ``xbar`` to avoid recalculating it:

    >>> m = mean(data)
    >>> variance(data, m)
    1.3720238095238095

    This function does not check that ``xbar`` is actually the mean of
    ``data``. Giving arbitrary values for ``xbar`` may lead to invalid or
    impossible results.

    Decimals and Fractions are supported:

    >>> from decimal import Decimal as D
    >>> variance([D("27.5"), D("30.25"), D("30.25"), D("34.5"), D("41.75")])
    Decimal('31.01875')

    >>> from fractions import Fraction as F
    >>> variance([F(1, 6), F(1, 2), F(5, 3)])
    Fraction(67, 108)

    """
    T, ss, c, n = _ss(data, xbar)
    if n < 2:
        raise StatisticsError('variance requires at least two data points')
    return _convert(ss / (n - 1), T)


def pvariance(data, mu=None):
    """Return the population variance of ``data``.

    data should be a sequence or iterable of Real-valued numbers, with at least one
    value. The optional argument mu, if given, should be the mean of
    the data. If it is missing or None, the mean is automatically calculated.

    Use this function to calculate the variance from the entire population.
    To estimate the variance from a sample, the ``variance`` function is
    usually a better choice.

    Examples:

    >>> data = [0.0, 0.25, 0.25, 1.25, 1.5, 1.75, 2.75, 3.25]
    >>> pvariance(data)
    1.25

    If you have already calculated the mean of the data, you can pass it as
    the optional second argument to avoid recalculating it:

    >>> mu = mean(data)
    >>> pvariance(data, mu)
    1.25

    Decimals and Fractions are supported:

    >>> from decimal import Decimal as D
    >>> pvariance([D("27.5"), D("30.25"), D("30.25"), D("34.5"), D("41.75")])
    Decimal('24.815')

    >>> from fractions import Fraction as F
    >>> pvariance([F(1, 4), F(5, 4), F(1, 2)])
    Fraction(13, 72)

    """
    T, ss, c, n = _ss(data, mu)
    if n < 1:
        raise StatisticsError('pvariance requires at least one data point')
    return _convert(ss / n, T)


def stdev(data, xbar=None):
    """Return the square root of the sample variance.

    See ``variance`` for arguments and other details.

    >>> stdev([1.5, 2.5, 2.5, 2.75, 3.25, 4.75])
    1.0810874155219827

    """
    T, ss, c, n = _ss(data, xbar)
    if n < 2:
        raise StatisticsError('stdev requires at least two data points')
    mss = ss / (n - 1)
    try:
        mss_numerator = mss.numerator
        mss_denominator = mss.denominator
    except AttributeError:
        raise ValueError('inf or nan encountered in data')
    if issubclass(T, Decimal):
        return _decimal_sqrt_of_frac(mss_numerator, mss_denominator)
    return _float_sqrt_of_frac(mss_numerator, mss_denominator)


def pstdev(data, mu=None):
    """Return the square root of the population variance.

    See ``pvariance`` for arguments and other details.

    >>> pstdev([1.5, 2.5, 2.5, 2.75, 3.25, 4.75])
    0.986893273527251

    """
    T, ss, c, n = _ss(data, mu)
    if n < 1:
        raise StatisticsError('pstdev requires at least one data point')
    mss = ss / n
    try:
        mss_numerator = mss.numerator
        mss_denominator = mss.denominator
    except AttributeError:
        raise ValueError('inf or nan encountered in data')
    if issubclass(T, Decimal):
        return _decimal_sqrt_of_frac(mss_numerator, mss_denominator)
    return _float_sqrt_of_frac(mss_numerator, mss_denominator)


def _mean_stdev(data):
    """In one pass, compute the mean and sample standard deviation as floats."""
    T, ss, xbar, n = _ss(data)
    if n < 2:
        raise StatisticsError('stdev requires at least two data points')
    mss = ss / (n - 1)
    try:
        return float(xbar), _float_sqrt_of_frac(mss.numerator, mss.denominator)
    except AttributeError:
        # Handle Nans and Infs gracefully
        return float(xbar), float(xbar) / float(ss)

def _sqrtprod(x: float, y: float) -> float:
    "Return sqrt(x * y) computed with improved accuracy and without overflow/underflow."
    h = sqrt(x * y)
    if not isfinite(h):
        if isinf(h) and not isinf(x) and not isinf(y):
            # Finite inputs overflowed, so scale down, and recompute.
            scale = 2.0 ** -512  # sqrt(1 / sys.float_info.max)
            return _sqrtprod(scale * x, scale * y) / scale
        return h
    if not h:
        if x and y:
            # Non-zero inputs underflowed, so scale up, and recompute.
            # Scale:  1 / sqrt(sys.float_info.min * sys.float_info.epsilon)
            scale = 2.0 ** 537
            return _sqrtprod(scale * x, scale * y) / scale
        return h
    # Improve accuracy with a differential correction.
    # https://www.wolframalpha.com/input/?i=Maclaurin+series+sqrt%28h**2+%2B+x%29+at+x%3D0
    d = sumprod((x, h), (y, -h))
    return h + d / (2.0 * h)


# === Statistics for relations between two inputs ===

# See https://en.wikipedia.org/wiki/Covariance
#     https://en.wikipedia.org/wiki/Pearson_correlation_coefficient
#     https://en.wikipedia.org/wiki/Simple_linear_regression


def covariance(x, y, /):
    """Covariance

    Return the sample covariance of two inputs *x* and *y*. Covariance
    is a measure of the joint variability of two inputs.

    >>> x = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    >>> y = [1, 2, 3, 1, 2, 3, 1, 2, 3]
    >>> covariance(x, y)
    0.75
    >>> z = [9, 8, 7, 6, 5, 4, 3, 2, 1]
    >>> covariance(x, z)
    -7.5
    >>> covariance(z, x)
    -7.5

    """
    n = len(x)
    if len(y) != n:
        raise StatisticsError('covariance requires that both inputs have same number of data points')
    if n < 2:
        raise StatisticsError('covariance requires at least two data points')
    xbar = fsum(x) / n
    ybar = fsum(y) / n
    sxy = sumprod((xi - xbar for xi in x), (yi - ybar for yi in y))
    return sxy / (n - 1)


def correlation(x, y, /, *, method='linear'):
    """Pearson's correlation coefficient

    Return the Pearson's correlation coefficient for two inputs. Pearson's
    correlation coefficient *r* takes values between -1 and +1. It measures
    the strength and direction of a linear relationship.

    >>> x = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    >>> y = [9, 8, 7, 6, 5, 4, 3, 2, 1]
    >>> correlation(x, x)
    1.0
    >>> correlation(x, y)
    -1.0

    If *method* is "ranked", computes Spearman's rank correlation coefficient
    for two inputs.  The data is replaced by ranks.  Ties are averaged
    so that equal values receive the same rank.  The resulting coefficient
    measures the strength of a monotonic relationship.

    Spearman's rank correlation coefficient is appropriate for ordinal
    data or for continuous data that doesn't meet the linear proportion
    requirement for Pearson's correlation coefficient.
    """
    n = len(x)
    if len(y) != n:
        raise StatisticsError('correlation requires that both inputs have same number of data points')
    if n < 2:
        raise StatisticsError('correlation requires at least two data points')
    if method not in {'linear', 'ranked'}:
        raise ValueError(f'Unknown method: {method!r}')
    if method == 'ranked':
        start = (n - 1) / -2            # Center rankings around zero
        x = _rank(x, start=start)
        y = _rank(y, start=start)
    else:
        xbar = fsum(x) / n
        ybar = fsum(y) / n
        x = [xi - xbar for xi in x]
        y = [yi - ybar for yi in y]
    sxy = sumprod(x, y)
    sxx = sumprod(x, x)
    syy = sumprod(y, y)
    try:
        return sxy / _sqrtprod(sxx, syy)
    except ZeroDivisionError:
        raise StatisticsError('at least one of the inputs is constant')


LinearRegression = namedtuple('LinearRegression', ('slope', 'intercept'))


def linear_regression(x, y, /, *, proportional=False):
    """Slope and intercept for simple linear regression.

    Return the slope and intercept of simple linear regression
    parameters estimated using ordinary least squares. Simple linear
    regression describes relationship between an independent variable
    *x* and a dependent variable *y* in terms of a linear function:

        y = slope * x + intercept + noise

    where *slope* and *intercept* are the regression parameters that are
    estimated, and noise represents the variability of the data that was
    not explained by the linear regression (it is equal to the
    difference between predicted and actual values of the dependent
    variable).

    The parameters are returned as a named tuple.

    >>> x = [1, 2, 3, 4, 5]
    >>> noise = NormalDist().samples(5, seed=42)
    >>> y = [3 * x[i] + 2 + noise[i] for i in range(5)]
    >>> linear_regression(x, y)  #doctest: +ELLIPSIS
    LinearRegression(slope=3.17495..., intercept=1.00925...)

    If *proportional* is true, the independent variable *x* and the
    dependent variable *y* are assumed to be directly proportional.
    The data is fit to a line passing through the origin.

    Since the *intercept* will always be 0.0, the underlying linear
    function simplifies to:

        y = slope * x + noise

    >>> y = [3 * x[i] + noise[i] for i in range(5)]
    >>> linear_regression(x, y, proportional=True)  #doctest: +ELLIPSIS
    LinearRegression(slope=2.90475..., intercept=0.0)

    """
    n = len(x)
    if len(y) != n:
        raise StatisticsError('linear regression requires that both inputs have same number of data points')
    if n < 2:
        raise StatisticsError('linear regression requires at least two data points')
    if not proportional:
        xbar = fsum(x) / n
        ybar = fsum(y) / n
        x = [xi - xbar for xi in x]  # List because used three times below
        y = (yi - ybar for yi in y)  # Generator because only used once below
    sxy = sumprod(x, y) + 0.0        # Add zero to coerce result to a float
    sxx = sumprod(x, x)
    try:
        slope = sxy / sxx   # equivalent to:  covariance(x, y) / variance(x)
    except ZeroDivisionError:
        raise StatisticsError('x is constant')
    intercept = 0.0 if proportional else ybar - slope * xbar
    return LinearRegression(slope=slope, intercept=intercept)


## Normal Distribution #####################################################


def _normal_dist_inv_cdf(p, mu, sigma):
    # There is no closed-form solution to the inverse CDF for the normal
    # distribution, so we use a rational approximation instead:
    # Wichura, M.J. (1988). "Algorithm AS241: The Percentage Points of the
    # Normal Distribution".  Applied Statistics. Blackwell Publishing. 37
    # (3): 477–484. doi:10.2307/2347330. JSTOR 2347330.
    q = p - 0.5
    if fabs(q) <= 0.425:
        r = 0.180625 - q * q
        # Hash sum: 55.88319_28806_14901_4439
        num = (((((((2.50908_09287_30122_6727e+3 * r +
                     3.34305_75583_58812_8105e+4) * r +
                     6.72657_70927_00870_0853e+4) * r +
                     4.59219_53931_54987_1457e+4) * r +
                     1.37316_93765_50946_1125e+4) * r +
                     1.97159_09503_06551_4427e+3) * r +
                     1.33141_66789_17843_7745e+2) * r +
                     3.38713_28727_96366_6080e+0) * q
        den = (((((((5.22649_52788_52854_5610e+3 * r +
                     2.87290_85735_72194_2674e+4) * r +
                     3.93078_95800_09271_0610e+4) * r +
                     2.12137_94301_58659_5867e+4) * r +
                     5.39419_60214_24751_1077e+3) * r +
                     6.87187_00749_20579_0830e+2) * r +
                     4.23133_30701_60091_1252e+1) * r +
                     1.0)
        x = num / den
        return mu + (x * sigma)
    r = p if q <= 0.0 else 1.0 - p
    r = sqrt(-log(r))
    if r <= 5.0:
        r = r - 1.6
        # Hash sum: 49.33206_50330_16102_89036
        num = (((((((7.74545_01427_83414_07640e-4 * r +
                     2.27238_44989_26918_45833e-2) * r +
                     2.41780_72517_74506_11770e-1) * r +
                     1.27045_82524_52368_38258e+0) * r +
                     3.64784_83247_63204_60504e+0) * r +
                     5.76949_72214_60691_40550e+0) * r +
                     4.63033_78461_56545_29590e+0) * r +
                     1.42343_71107_49683_57734e+0)
        den = (((((((1.05075_00716_44416_84324e-9 * r +
                     5.47593_80849_95344_94600e-4) * r +
                     1.51986_66563_61645_71966e-2) * r +
                     1.48103_97642_74800_74590e-1) * r +
                     6.89767_33498_51000_04550e-1) * r +
                     1.67638_48301_83803_84940e+0) * r +
                     2.05319_16266_37758_82187e+0) * r +
                     1.0)
    else:
        r = r - 5.0
        # Hash sum: 47.52583_31754_92896_71629
        num = (((((((2.01033_43992_92288_13265e-7 * r +
                     2.71155_55687_43487_57815e-5) * r +
                     1.24266_09473_88078_43860e-3) * r +
                     2.65321_89526_57612_30930e-2) * r +
                     2.96560_57182_85048_91230e-1) * r +
                     1.78482_65399_17291_33580e+0) * r +
                     5.46378_49111_64114_36990e+0) * r +
                     6.65790_46435_01103_77720e+0)
        den = (((((((2.04426_31033_89939_78564e-15 * r +
                     1.42151_17583_16445_88870e-7) * r +
                     1.84631_83175_10054_68180e-5) * r +
                     7.86869_13114_56132_59100e-4) * r +
                     1.48753_61290_85061_48525e-2) * r +
                     1.36929_88092_27358_05310e-1) * r +
                     5.99832_20655_58879_37690e-1) * r +
                     1.0)
    x = num / den
    if q < 0.0:
        x = -x
    return mu + (x * sigma)


# If available, use C implementation
try:
    from _statistics import _normal_dist_inv_cdf
except ImportError:
    pass


class NormalDist:
    "Normal distribution of a random variable"
    # https://en.wikipedia.org/wiki/Normal_distribution
    # https://en.wikipedia.org/wiki/Variance#Properties

    __slots__ = {
        '_mu': 'Arithmetic mean of a normal distribution',
        '_sigma': 'Standard deviation of a normal distribution',
    }

    def __init__(self, mu=0.0, sigma=1.0):
        "NormalDist where mu is the mean and sigma is the standard deviation."
        if sigma < 0.0:
            raise StatisticsError('sigma must be non-negative')
        self._mu = float(mu)
        self._sigma = float(sigma)

    @classmethod
    def from_samples(cls, data):
        "Make a normal distribution instance from sample data."
        return cls(*_mean_stdev(data))

    def samples(self, n, *, seed=None):
        "Generate *n* samples for a given mean and standard deviation."
        rnd = random.random if seed is None else random.Random(seed).random
        inv_cdf = _normal_dist_inv_cdf
        mu = self._mu
        sigma = self._sigma
        return [inv_cdf(rnd(), mu, sigma) for _ in repeat(None, n)]

    def pdf(self, x):
        "Probability density function.  P(x <= X < x+dx) / dx"
        variance = self._sigma * self._sigma
        if not variance:
            raise StatisticsError('pdf() not defined when sigma is zero')
        diff = x - self._mu
        return exp(diff * diff / (-2.0 * variance)) / sqrt(tau * variance)

    def cdf(self, x):
        "Cumulative distribution function.  P(X <= x)"
        if not self._sigma:
            raise StatisticsError('cdf() not defined when sigma is zero')
        return 0.5 * (1.0 + erf((x - self._mu) / (self._sigma * _SQRT2)))

    def inv_cdf(self, p):
        """Inverse cumulative distribution function.  x : P(X <= x) = p

        Finds the value of the random variable such that the probability of
        the variable being less than or equal to that value equals the given
        probability.

        This function is also called the percent point function or quantile
        function.
        """
        if p <= 0.0 or p >= 1.0:
            raise StatisticsError('p must be in the range 0.0 < p < 1.0')
        return _normal_dist_inv_cdf(p, self._mu, self._sigma)

    def quantiles(self, n=4):
        """Divide into *n* continuous intervals with equal probability.

        Returns a list of (n - 1) cut points separating the intervals.

        Set *n* to 4 for quartiles (the default).  Set *n* to 10 for deciles.
        Set *n* to 100 for percentiles which gives the 99 cuts points that
        separate the normal distribution in to 100 equal sized groups.
        """
        return [self.inv_cdf(i / n) for i in range(1, n)]

    def overlap(self, other):
        """Compute the overlapping coefficient (OVL) between two normal distributions.

        Measures the agreement between two normal probability distributions.
        Returns a value between 0.0 and 1.0 giving the overlapping area in
        the two underlying probability density functions.

            >>> N1 = NormalDist(2.4, 1.6)
            >>> N2 = NormalDist(3.2, 2.0)
            >>> N1.overlap(N2)
            0.8035050657330205
        """
        # See: "The overlapping coefficient as a measure of agreement between
        # probability distributions and point estimation of the overlap of two
        # normal densities" -- Henry F. Inman and Edwin L. Bradley Jr
        # http://dx.doi.org/10.1080/03610928908830127
        if not isinstance(other, NormalDist):
            raise TypeError('Expected another NormalDist instance')
        X, Y = self, other
        if (Y._sigma, Y._mu) < (X._sigma, X._mu):  # sort to assure commutativity
            X, Y = Y, X
        X_var, Y_var = X.variance, Y.variance
        if not X_var or not Y_var:
            raise StatisticsError('overlap() not defined when sigma is zero')
        dv = Y_var - X_var
        dm = fabs(Y._mu - X._mu)
        if not dv:
            return 1.0 - erf(dm / (2.0 * X._sigma * _SQRT2))
        a = X._mu * Y_var - Y._mu * X_var
        b = X._sigma * Y._sigma * sqrt(dm * dm + dv * log(Y_var / X_var))
        x1 = (a + b) / dv
        x2 = (a - b) / dv
        return 1.0 - (fabs(Y.cdf(x1) - X.cdf(x1)) + fabs(Y.cdf(x2) - X.cdf(x2)))

    def zscore(self, x):
        """Compute the Standard Score.  (x - mean) / stdev

        Describes *x* in terms of the number of standard deviations
        above or below the mean of the normal distribution.
        """
        # https://www.statisticshowto.com/probability-and-statistics/z-score/
        if not self._sigma:
            raise StatisticsError('zscore() not defined when sigma is zero')
        return (x - self._mu) / self._sigma

    @property
    def mean(self):
        "Arithmetic mean of the normal distribution."
        return self._mu

    @property
    def median(self):
        "Return the median of the normal distribution"
        return self._mu

    @property
    def mode(self):
        """Return the mode of the normal distribution

        The mode is the value x where which the probability density
        function (pdf) takes its maximum value.
        """
        return self._mu

    @property
    def stdev(self):
        "Standard deviation of the normal distribution."
        return self._sigma

    @property
    def variance(self):
        "Square of the standard deviation."
        return self._sigma * self._sigma

    def __add__(x1, x2):
        """Add a constant or another NormalDist instance.

        If *other* is a constant, translate mu by the constant,
        leaving sigma unchanged.

        If *other* is a NormalDist, add both the means and the variances.
        Mathematically, this works only if the two distributions are
        independent or if they are jointly normally distributed.
        """
        if isinstance(x2, NormalDist):
            return NormalDist(x1._mu + x2._mu, hypot(x1._sigma, x2._sigma))
        return NormalDist(x1._mu + x2, x1._sigma)

    def __sub__(x1, x2):
        """Subtract a constant or another NormalDist instance.

        If *other* is a constant, translate by the constant mu,
        leaving sigma unchanged.

        If *other* is a NormalDist, subtract the means and add the variances.
        Mathematically, this works only if the two distributions are
        independent or if they are jointly normally distributed.
        """
        if isinstance(x2, NormalDist):
            return NormalDist(x1._mu - x2._mu, hypot(x1._sigma, x2._sigma))
        return NormalDist(x1._mu - x2, x1._sigma)

    def __mul__(x1, x2):
        """Multiply both mu and sigma by a constant.

        Used for rescaling, perhaps to change measurement units.
        Sigma is scaled with the absolute value of the constant.
        """
        return NormalDist(x1._mu * x2, x1._sigma * fabs(x2))

    def __truediv__(x1, x2):
        """Divide both mu and sigma by a constant.

        Used for rescaling, perhaps to change measurement units.
        Sigma is scaled with the absolute value of the constant.
        """
        return NormalDist(x1._mu / x2, x1._sigma / fabs(x2))

    def __pos__(x1):
        "Return a copy of the instance."
        return NormalDist(x1._mu, x1._sigma)

    def __neg__(x1):
        "Negates mu while keeping sigma the same."
        return NormalDist(-x1._mu, x1._sigma)

    __radd__ = __add__

    def __rsub__(x1, x2):
        "Subtract a NormalDist from a constant or another NormalDist."
        return -(x1 - x2)

    __rmul__ = __mul__

    def __eq__(x1, x2):
        "Two NormalDist objects are equal if their mu and sigma are both equal."
        if not isinstance(x2, NormalDist):
            return NotImplemented
        return x1._mu == x2._mu and x1._sigma == x2._sigma

    def __hash__(self):
        "NormalDist objects hash equal if their mu and sigma are both equal."
        return hash((self._mu, self._sigma))

    def __repr__(self):
        return f'{type(self).__name__}(mu={self._mu!r}, sigma={self._sigma!r})'

    def __getstate__(self):
        return self._mu, self._sigma

    def __setstate__(self, state):
        self._mu, self._sigma = state


## kde_random() ##############################################################

def _newton_raphson(f_inv_estimate, f, f_prime, tolerance=1e-12):
    def f_inv(y):
        "Return x such that f(x) ≈ y within the specified tolerance."
        x = f_inv_estimate(y)
        while abs(diff := f(x) - y) > tolerance:
            x -= diff / f_prime(x)
        return x
    return f_inv

def _quartic_invcdf_estimate(p):
    sign, p = (1.0, p) if p <= 1/2 else (-1.0, 1.0 - p)
    x = (2.0 * p) ** 0.4258865685331 - 1.0
    if p >= 0.004 < 0.499:
        x += 0.026818732 * sin(7.101753784 * p + 2.73230839482953)
    return x * sign

_quartic_invcdf = _newton_raphson(
    f_inv_estimate = _quartic_invcdf_estimate,
    f = lambda t: 3/16 * t**5 - 5/8 * t**3 + 15/16 * t + 1/2,
    f_prime = lambda t: 15/16 * (1.0 - t * t) ** 2)

def _triweight_invcdf_estimate(p):
    sign, p = (1.0, p) if p <= 1/2 else (-1.0, 1.0 - p)
    x = (2.0 * p) ** 0.3400218741872791 - 1.0
    return x * sign

_triweight_invcdf = _newton_raphson(
    f_inv_estimate = _triweight_invcdf_estimate,
    f = lambda t: 35/32 * (-1/7*t**7 + 3/5*t**5 - t**3 + t) + 1/2,
    f_prime = lambda t: 35/32 * (1.0 - t * t) ** 3)

_kernel_invcdfs = {
    'normal': NormalDist().inv_cdf,
    'logistic': lambda p: log(p / (1 - p)),
    'sigmoid': lambda p: log(tan(p * pi/2)),
    'rectangular': lambda p: 2*p - 1,
    'parabolic': lambda p: 2 * cos((acos(2*p-1) + pi) / 3),
    'quartic': _quartic_invcdf,
    'triweight': _triweight_invcdf,
    'triangular': lambda p: sqrt(2*p) - 1 if p < 1/2 else 1 - sqrt(2 - 2*p),
    'cosine': lambda p: 2 * asin(2*p - 1) / pi,
}
_kernel_invcdfs['gauss'] = _kernel_invcdfs['normal']
_kernel_invcdfs['uniform'] = _kernel_invcdfs['rectangular']
_kernel_invcdfs['epanechnikov'] = _kernel_invcdfs['parabolic']
_kernel_invcdfs['biweight'] = _kernel_invcdfs['quartic']

def kde_random(data, h, kernel='normal', *, seed=None):
    """Return a function that makes a random selection from the estimated
    probability density function created by kde(data, h, kernel).

    Providing a *seed* allows reproducible selections within a single
    thread.  The seed may be an integer, float, str, or bytes.

    A StatisticsError will be raised if the *data* sequence is empty.

    Example:

    >>> data = [-2.1, -1.3, -0.4, 1.9, 5.1, 6.2]
    >>> rand = kde_random(data, h=1.5, seed=8675309)
    >>> new_selections = [rand() for i in range(10)]
    >>> [round(x, 1) for x in new_selections]
    [0.7, 6.2, 1.2, 6.9, 7.0, 1.8, 2.5, -0.5, -1.8, 5.6]

    """
    n = len(data)
    if not n:
        raise StatisticsError('Empty data sequence')

    if not isinstance(data[0], (int, float)):
        raise TypeError('Data sequence must contain ints or floats')

    if h <= 0.0:
        raise StatisticsError(f'Bandwidth h must be positive, not {h=!r}')

    kernel_invcdf = _kernel_invcdfs.get(kernel)
    if kernel_invcdf is None:
        raise StatisticsError(f'Unknown kernel name: {kernel!r}')

    prng = _random.Random(seed)
    random = prng.random
    choice = prng.choice

    def rand():
        return choice(data) + h * kernel_invcdf(random())

    rand.__doc__ = f'Random KDE selection with {h=!r} and {kernel=!r}'

    return rand



"""Generate cryptographically strong pseudo-random numbers suitable for
managing secrets such as account authentication, tokens, and similar.

See PEP 506 for more information.
https://peps.python.org/pep-0506/

"""

__all__ = ['choice', 'randbelow', 'randbits', 'SystemRandom',
           'token_bytes', 'token_hex', 'token_urlsafe',
           'compare_digest',
           ]


import base64

from hmac import compare_digest
from random import SystemRandom

_sysrand = SystemRandom()

randbits = _sysrand.getrandbits
choice = _sysrand.choice

def randbelow(exclusive_upper_bound):
    """Return a random int in the range [0, n)."""
    if exclusive_upper_bound <= 0:
        raise ValueError("Upper bound must be positive.")
    return _sysrand._randbelow(exclusive_upper_bound)

DEFAULT_ENTROPY = 32  # number of bytes to return by default

def token_bytes(nbytes=None):
    """Return a random byte string containing *nbytes* bytes.

    If *nbytes* is ``None`` or not supplied, a reasonable
    default is used.

    >>> token_bytes(16)  #doctest:+SKIP
    b'\\xebr\\x17D*t\\xae\\xd4\\xe3S\\xb6\\xe2\\xebP1\\x8b'

    """
    if nbytes is None:
        nbytes = DEFAULT_ENTROPY
    return _sysrand.randbytes(nbytes)

def token_hex(nbytes=None):
    """Return a random text string, in hexadecimal.

    The string has *nbytes* random bytes, each byte converted to two
    hex digits.  If *nbytes* is ``None`` or not supplied, a reasonable
    default is used.

    >>> token_hex(16)  #doctest:+SKIP
    'f9bf78b9a18ce6d46a0cd2b0b86df9da'

    """
    return token_bytes(nbytes).hex()

def token_urlsafe(nbytes=None):
    """Return a random URL-safe text string, in Base64 encoding.

    The string has *nbytes* random bytes.  If *nbytes* is ``None``
    or not supplied, a reasonable default is used.

    >>> token_urlsafe(16)  #doctest:+SKIP
    'Drmhze6EPcv0fN_81Bj-nA'

    """
    tok = token_bytes(nbytes)
    return base64.urlsafe_b64encode(tok).rstrip(b'=').decode('ascii')



import sys
import builtins as bltns
from functools import partial
from types import MappingProxyType, DynamicClassAttribute


__all__ = [
        'EnumType', 'EnumMeta', 'EnumDict',
        'Enum', 'IntEnum', 'StrEnum', 'Flag', 'IntFlag', 'ReprEnum',
        'auto', 'unique', 'property', 'verify', 'member', 'nonmember',
        'FlagBoundary', 'STRICT', 'CONFORM', 'EJECT', 'KEEP',
        'global_flag_repr', 'global_enum_repr', 'global_str', 'global_enum',
        'EnumCheck', 'CONTINUOUS', 'NAMED_FLAGS', 'UNIQUE',
        'pickle_by_global_name', 'pickle_by_enum_name',
        ]


# Dummy value for Enum and Flag as there are explicit checks for them
# before they have been created.
# This is also why there are checks in EnumType like `if Enum is not None`
Enum = Flag = EJECT = _stdlib_enums = ReprEnum = None

class nonmember(object):
    """
    Protects item from becoming an Enum member during class creation.
    """
    def __init__(self, value):
        self.value = value

class member(object):
    """
    Forces item to become an Enum member during class creation.
    """
    def __init__(self, value):
        self.value = value

def _is_descriptor(obj):
    """
    Returns True if obj is a descriptor, False otherwise.
    """
    return not isinstance(obj, partial) and (
            hasattr(obj, '__get__') or
            hasattr(obj, '__set__') or
            hasattr(obj, '__delete__')
            )

def _is_dunder(name):
    """
    Returns True if a __dunder__ name, False otherwise.
    """
    return (
            len(name) > 4 and
            name[:2] == name[-2:] == '__' and
            name[2] != '_' and
            name[-3] != '_'
            )

def _is_sunder(name):
    """
    Returns True if a _sunder_ name, False otherwise.
    """
    return (
            len(name) > 2 and
            name[0] == name[-1] == '_' and
            name[1] != '_' and
            name[-2] != '_'
            )

def _is_internal_class(cls_name, obj):
    # do not use `re` as `re` imports `enum`
    if not isinstance(obj, type):
        return False
    qualname = getattr(obj, '__qualname__', '')
    s_pattern = cls_name + '.' + getattr(obj, '__name__', '')
    e_pattern = '.' + s_pattern
    return qualname == s_pattern or qualname.endswith(e_pattern)

def _is_private(cls_name, name):
    # do not use `re` as `re` imports `enum`
    pattern = '_%s__' % (cls_name, )
    pat_len = len(pattern)
    if (
            len(name) > pat_len
            and name.startswith(pattern)
            and (name[-1] != '_' or name[-2] != '_')
        ):
        return True
    else:
        return False

def _is_single_bit(num):
    """
    True if only one bit set in num (should be an int)
    """
    if num == 0:
        return False
    num &= num - 1
    return num == 0

def _make_class_unpicklable(obj):
    """
    Make the given obj un-picklable.

    obj should be either a dictionary, or an Enum
    """
    def _break_on_call_reduce(self, proto):
        raise TypeError('%r cannot be pickled' % self)
    if isinstance(obj, dict):
        obj['__reduce_ex__'] = _break_on_call_reduce
        obj['__module__'] = '<unknown>'
    else:
        setattr(obj, '__reduce_ex__', _break_on_call_reduce)
        setattr(obj, '__module__', '<unknown>')

def _iter_bits_lsb(num):
    # num must be a positive integer
    original = num
    if isinstance(num, Enum):
        num = num.value
    if num < 0:
        raise ValueError('%r is not a positive integer' % original)
    while num:
        b = num & (~num + 1)
        yield b
        num ^= b

def show_flag_values(value):
    return list(_iter_bits_lsb(value))

def bin(num, max_bits=None):
    """
    Like built-in bin(), except negative values are represented in
    twos-complement, and the leading bit always indicates sign
    (0=positive, 1=negative).

    >>> bin(10)
    '0b0 1010'
    >>> bin(~10)   # ~10 is -11
    '0b1 0101'
    """

    num = num.__index__()
    ceiling = 2 ** (num).bit_length()
    if num >= 0:
        s = bltns.bin(num + ceiling).replace('1', '0', 1)
    else:
        s = bltns.bin(~num ^ (ceiling - 1) + ceiling)
    sign = s[:3]
    digits = s[3:]
    if max_bits is not None:
        if len(digits) < max_bits:
            digits = (sign[-1] * max_bits + digits)[-max_bits:]
    return "%s %s" % (sign, digits)

def _dedent(text):
    """
    Like textwrap.dedent.  Rewritten because we cannot import textwrap.
    """
    lines = text.split('\n')
    for i, ch in enumerate(lines[0]):
        if ch != ' ':
            break
    for j, l in enumerate(lines):
        lines[j] = l[i:]
    return '\n'.join(lines)

class _not_given:
    def __repr__(self):
        return('<not given>')
_not_given = _not_given()

class _auto_null:
    def __repr__(self):
        return '_auto_null'
_auto_null = _auto_null()

class auto:
    """
    Instances are replaced with an appropriate value in Enum class suites.
    """
    def __init__(self, value=_auto_null):
        self.value = value

    def __repr__(self):
        return "auto(%r)" % self.value

class property(DynamicClassAttribute):
    """
    This is a descriptor, used to define attributes that act differently
    when accessed through an enum member and through an enum class.
    Instance access is the same as property(), but access to an attribute
    through the enum class will instead look in the class' _member_map_ for
    a corresponding enum member.
    """

    member = None
    _attr_type = None
    _cls_type = None

    def __get__(self, instance, ownerclass=None):
        if instance is None:
            if self.member is not None:
                return self.member
            else:
                raise AttributeError(
                        '%r has no attribute %r' % (ownerclass, self.name)
                        )
        if self.fget is not None:
            # use previous enum.property
            return self.fget(instance)
        elif self._attr_type == 'attr':
            # look up previous attibute
            return getattr(self._cls_type, self.name)
        elif self._attr_type == 'desc':
            # use previous descriptor
            return getattr(instance._value_, self.name)
        # look for a member by this name.
        try:
            return ownerclass._member_map_[self.name]
        except KeyError:
            raise AttributeError(
                    '%r has no attribute %r' % (ownerclass, self.name)
                    ) from None

    def __set__(self, instance, value):
        if self.fset is not None:
            return self.fset(instance, value)
        raise AttributeError(
                "<enum %r> cannot set attribute %r" % (self.clsname, self.name)
                )

    def __delete__(self, instance):
        if self.fdel is not None:
            return self.fdel(instance)
        raise AttributeError(
                "<enum %r> cannot delete attribute %r" % (self.clsname, self.name)
                )

    def __set_name__(self, ownerclass, name):
        self.name = name
        self.clsname = ownerclass.__name__


class _proto_member:
    """
    intermediate step for enum members between class execution and final creation
    """

    def __init__(self, value):
        self.value = value

    def __set_name__(self, enum_class, member_name):
        """
        convert each quasi-member into an instance of the new enum class
        """
        # first step: remove ourself from enum_class
        delattr(enum_class, member_name)
        # second step: create member based on enum_class
        value = self.value
        if not isinstance(value, tuple):
            args = (value, )
        else:
            args = value
        if enum_class._member_type_ is tuple:   # special case for tuple enums
            args = (args, )     # wrap it one more time
        if not enum_class._use_args_:
            enum_member = enum_class._new_member_(enum_class)
        else:
            enum_member = enum_class._new_member_(enum_class, *args)
        if not hasattr(enum_member, '_value_'):
            if enum_class._member_type_ is object:
                enum_member._value_ = value
            else:
                try:
                    enum_member._value_ = enum_class._member_type_(*args)
                except Exception as exc:
                    new_exc = TypeError(
                            '_value_ not set in __new__, unable to create it'
                            )
                    new_exc.__cause__ = exc
                    raise new_exc
        value = enum_member._value_
        enum_member._name_ = member_name
        enum_member.__objclass__ = enum_class
        enum_member.__init__(*args)
        enum_member._sort_order_ = len(enum_class._member_names_)

        if Flag is not None and issubclass(enum_class, Flag):
            if isinstance(value, int):
                enum_class._flag_mask_ |= value
                if _is_single_bit(value):
                    enum_class._singles_mask_ |= value
            enum_class._all_bits_ = 2 ** ((enum_class._flag_mask_).bit_length()) - 1

        # If another member with the same value was already defined, the
        # new member becomes an alias to the existing one.
        try:
            try:
                # try to do a fast lookup to avoid the quadratic loop
                enum_member = enum_class._value2member_map_[value]
            except TypeError:
                for name, canonical_member in enum_class._member_map_.items():
                    if canonical_member._value_ == value:
                        enum_member = canonical_member
                        break
                else:
                    raise KeyError
        except KeyError:
            # this could still be an alias if the value is multi-bit and the
            # class is a flag class
            if (
                    Flag is None
                    or not issubclass(enum_class, Flag)
                ):
                # no other instances found, record this member in _member_names_
                enum_class._member_names_.append(member_name)
            elif (
                    Flag is not None
                    and issubclass(enum_class, Flag)
                    and isinstance(value, int)
                    and _is_single_bit(value)
                ):
                # no other instances found, record this member in _member_names_
                enum_class._member_names_.append(member_name)

        enum_class._add_member_(member_name, enum_member)
        try:
            # This may fail if value is not hashable. We can't add the value
            # to the map, and by-value lookups for this value will be
            # linear.
            enum_class._value2member_map_.setdefault(value, enum_member)
            if value not in enum_class._hashable_values_:
                enum_class._hashable_values_.append(value)
        except TypeError:
            # keep track of the value in a list so containment checks are quick
            enum_class._unhashable_values_.append(value)
            enum_class._unhashable_values_map_.setdefault(member_name, []).append(value)


class EnumDict(dict):
    """
    Track enum member order and ensure member names are not reused.

    EnumType will use the names found in self._member_names as the
    enumeration member names.
    """
    def __init__(self, cls_name=None):
        super().__init__()
        self._member_names = {} # use a dict -- faster look-up than a list, and keeps insertion order since 3.7
        self._last_values = []
        self._ignore = []
        self._auto_called = False
        self._cls_name = cls_name

    def __setitem__(self, key, value):
        """
        Changes anything not dundered or not a descriptor.

        If an enum member name is used twice, an error is raised; duplicate
        values are not checked for.

        Single underscore (sunder) names are reserved.
        """
        if self._cls_name is not None and _is_private(self._cls_name, key):
            # do nothing, name will be a normal attribute
            pass
        elif _is_sunder(key):
            if key not in (
                    '_order_',
                    '_generate_next_value_', '_numeric_repr_', '_missing_', '_ignore_',
                    '_iter_member_', '_iter_member_by_value_', '_iter_member_by_def_',
                    '_add_alias_', '_add_value_alias_',
                    # While not in use internally, those are common for pretty
                    # printing and thus excluded from Enum's reservation of
                    # _sunder_ names
                    ) and not key.startswith('_repr_'):
                raise ValueError(
                        '_sunder_ names, such as %r, are reserved for future Enum use'
                        % (key, )
                        )
            if key == '_generate_next_value_':
                # check if members already defined as auto()
                if self._auto_called:
                    raise TypeError("_generate_next_value_ must be defined before members")
                _gnv = value.__func__ if isinstance(value, staticmethod) else value
                setattr(self, '_generate_next_value', _gnv)
            elif key == '_ignore_':
                if isinstance(value, str):
                    value = value.replace(',',' ').split()
                else:
                    value = list(value)
                self._ignore = value
                already = set(value) & set(self._member_names)
                if already:
                    raise ValueError(
                            '_ignore_ cannot specify already set names: %r'
                            % (already, )
                            )
        elif _is_dunder(key):
            if key == '__order__':
                key = '_order_'
        elif key in self._member_names:
            # descriptor overwriting an enum?
            raise TypeError('%r already defined as %r' % (key, self[key]))
        elif key in self._ignore:
            pass
        elif isinstance(value, nonmember):
            # unwrap value here; it won't be processed by the below `else`
            value = value.value
        elif isinstance(value, partial):
            import warnings
            warnings.warn('functools.partial will be a method descriptor '
                          'in future Python versions; wrap it in '
                          'enum.member() if you want to preserve the '
                          'old behavior', FutureWarning, stacklevel=2)
        elif _is_descriptor(value):
            pass
        elif self._cls_name is not None and _is_internal_class(self._cls_name, value):
            # do nothing, name will be a normal attribute
            pass
        else:
            if key in self:
                # enum overwriting a descriptor?
                raise TypeError('%r already defined as %r' % (key, self[key]))
            elif isinstance(value, member):
                # unwrap value here -- it will become a member
                value = value.value
            non_auto_store = True
            single = False
            if isinstance(value, auto):
                single = True
                value = (value, )
            if isinstance(value, tuple) and any(isinstance(v, auto) for v in value):
                # insist on an actual tuple, no subclasses, in keeping with only supporting
                # top-level auto() usage (not contained in any other data structure)
                auto_valued = []
                t = type(value)
                for v in value:
                    if isinstance(v, auto):
                        non_auto_store = False
                        if v.value == _auto_null:
                            v.value = self._generate_next_value(
                                    key, 1, len(self._member_names), self._last_values[:],
                                    )
                            self._auto_called = True
                        v = v.value
                        self._last_values.append(v)
                    auto_valued.append(v)
                if single:
                    value = auto_valued[0]
                else:
                    try:
                        # accepts iterable as multiple arguments?
                        value = t(auto_valued)
                    except TypeError:
                        # then pass them in singly
                        value = t(*auto_valued)
            self._member_names[key] = None
            if non_auto_store:
                self._last_values.append(value)
        super().__setitem__(key, value)

    @property
    def member_names(self):
        return list(self._member_names)

    def update(self, members, **more_members):
        try:
            for name in members.keys():
                self[name] = members[name]
        except AttributeError:
            for name, value in members:
                self[name] = value
        for name, value in more_members.items():
            self[name] = value

_EnumDict = EnumDict        # keep private name for backwards compatibility


class EnumType(type):
    """
    Metaclass for Enum
    """

    @classmethod
    def __prepare__(metacls, cls, bases, **kwds):
        # check that previous enum members do not exist
        metacls._check_for_existing_members_(cls, bases)
        # create the namespace dict
        enum_dict = EnumDict(cls)
        # inherit previous flags and _generate_next_value_ function
        member_type, first_enum = metacls._get_mixins_(cls, bases)
        if first_enum is not None:
            enum_dict['_generate_next_value_'] = getattr(
                    first_enum, '_generate_next_value_', None,
                    )
        return enum_dict

    def __new__(metacls, cls, bases, classdict, *, boundary=None, _simple=False, **kwds):
        # an Enum class is final once enumeration items have been defined; it
        # cannot be mixed with other types (int, float, etc.) if it has an
        # inherited __new__ unless a new __new__ is defined (or the resulting
        # class will fail).
        #
        if _simple:
            return super().__new__(metacls, cls, bases, classdict, **kwds)
        #
        # remove any keys listed in _ignore_
        classdict.setdefault('_ignore_', []).append('_ignore_')
        ignore = classdict['_ignore_']
        for key in ignore:
            classdict.pop(key, None)
        #
        # grab member names
        member_names = classdict._member_names
        #
        # check for illegal enum names (any others?)
        invalid_names = set(member_names) & {'mro', ''}
        if invalid_names:
            raise ValueError('invalid enum member name(s) %s'  % (
                    ','.join(repr(n) for n in invalid_names)
                    ))
        #
        # adjust the sunders
        _order_ = classdict.pop('_order_', None)
        _gnv = classdict.get('_generate_next_value_')
        if _gnv is not None and type(_gnv) is not staticmethod:
            _gnv = staticmethod(_gnv)
        # convert to normal dict
        classdict = dict(classdict.items())
        if _gnv is not None:
            classdict['_generate_next_value_'] = _gnv
        #
        # data type of member and the controlling Enum class
        member_type, first_enum = metacls._get_mixins_(cls, bases)
        __new__, save_new, use_args = metacls._find_new_(
                classdict, member_type, first_enum,
                )
        classdict['_new_member_'] = __new__
        classdict['_use_args_'] = use_args
        #
        # convert future enum members into temporary _proto_members
        for name in member_names:
            value = classdict[name]
            classdict[name] = _proto_member(value)
        #
        # house-keeping structures
        classdict['_member_names_'] = []
        classdict['_member_map_'] = {}
        classdict['_value2member_map_'] = {}
        classdict['_hashable_values_'] = []          # for comparing with non-hashable types
        classdict['_unhashable_values_'] = []       # e.g. frozenset() with set()
        classdict['_unhashable_values_map_'] = {}
        classdict['_member_type_'] = member_type
        # now set the __repr__ for the value
        classdict['_value_repr_'] = metacls._find_data_repr_(cls, bases)
        #
        # Flag structures (will be removed if final class is not a Flag
        classdict['_boundary_'] = (
                boundary
                or getattr(first_enum, '_boundary_', None)
                )
        classdict['_flag_mask_'] = 0
        classdict['_singles_mask_'] = 0
        classdict['_all_bits_'] = 0
        classdict['_inverted_'] = None
        try:
            classdict['_%s__in_progress' % cls] = True
            enum_class = super().__new__(metacls, cls, bases, classdict, **kwds)
            classdict['_%s__in_progress' % cls] = False
            delattr(enum_class, '_%s__in_progress' % cls)
        except Exception as e:
            # since 3.12 the note "Error calling __set_name__ on '_proto_member' instance ..."
            # is tacked on to the error instead of raising a RuntimeError, so discard it
            if hasattr(e, '__notes__'):
                del e.__notes__
            raise
        # update classdict with any changes made by __init_subclass__
        classdict.update(enum_class.__dict__)
        #
        # double check that repr and friends are not the mixin's or various
        # things break (such as pickle)
        # however, if the method is defined in the Enum itself, don't replace
        # it
        #
        # Also, special handling for ReprEnum
        if ReprEnum is not None and ReprEnum in bases:
            if member_type is object:
                raise TypeError(
                        'ReprEnum subclasses must be mixed with a data type (i.e.'
                        ' int, str, float, etc.)'
                        )
            if '__format__' not in classdict:
                enum_class.__format__ = member_type.__format__
                classdict['__format__'] = enum_class.__format__
            if '__str__' not in classdict:
                method = member_type.__str__
                if method is object.__str__:
                    # if member_type does not define __str__, object.__str__ will use
                    # its __repr__ instead, so we'll also use its __repr__
                    method = member_type.__repr__
                enum_class.__str__ = method
                classdict['__str__'] = enum_class.__str__
        for name in ('__repr__', '__str__', '__format__', '__reduce_ex__'):
            if name not in classdict:
                # check for mixin overrides before replacing
                enum_method = getattr(first_enum, name)
                found_method = getattr(enum_class, name)
                object_method = getattr(object, name)
                data_type_method = getattr(member_type, name)
                if found_method in (data_type_method, object_method):
                    setattr(enum_class, name, enum_method)
        #
        # for Flag, add __or__, __and__, __xor__, and __invert__
        if Flag is not None and issubclass(enum_class, Flag):
            for name in (
                    '__or__', '__and__', '__xor__',
                    '__ror__', '__rand__', '__rxor__',
                    '__invert__'
                ):
                if name not in classdict:
                    enum_method = getattr(Flag, name)
                    setattr(enum_class, name, enum_method)
                    classdict[name] = enum_method
        #
        # replace any other __new__ with our own (as long as Enum is not None,
        # anyway) -- again, this is to support pickle
        if Enum is not None:
            # if the user defined their own __new__, save it before it gets
            # clobbered in case they subclass later
            if save_new:
                enum_class.__new_member__ = __new__
            enum_class.__new__ = Enum.__new__
        #
        # py3 support for definition order (helps keep py2/py3 code in sync)
        #
        # _order_ checking is spread out into three/four steps
        # - if enum_class is a Flag:
        #   - remove any non-single-bit flags from _order_
        # - remove any aliases from _order_
        # - check that _order_ and _member_names_ match
        #
        # step 1: ensure we have a list
        if _order_ is not None:
            if isinstance(_order_, str):
                _order_ = _order_.replace(',', ' ').split()
        #
        # remove Flag structures if final class is not a Flag
        if (
                Flag is None and cls != 'Flag'
                or Flag is not None and not issubclass(enum_class, Flag)
            ):
            delattr(enum_class, '_boundary_')
            delattr(enum_class, '_flag_mask_')
            delattr(enum_class, '_singles_mask_')
            delattr(enum_class, '_all_bits_')
            delattr(enum_class, '_inverted_')
        elif Flag is not None and issubclass(enum_class, Flag):
            # set correct __iter__
            member_list = [m._value_ for m in enum_class]
            if member_list != sorted(member_list):
                enum_class._iter_member_ = enum_class._iter_member_by_def_
            if _order_:
                # _order_ step 2: remove any items from _order_ that are not single-bit
                _order_ = [
                        o
                        for o in _order_
                        if o not in enum_class._member_map_ or _is_single_bit(enum_class[o]._value_)
                        ]
        #
        if _order_:
            # _order_ step 3: remove aliases from _order_
            _order_ = [
                    o
                    for o in _order_
                    if (
                        o not in enum_class._member_map_
                        or
                        (o in enum_class._member_map_ and o in enum_class._member_names_)
                        )]
            # _order_ step 4: verify that _order_ and _member_names_ match
            if _order_ != enum_class._member_names_:
                raise TypeError(
                        'member order does not match _order_:\n  %r\n  %r'
                        % (enum_class._member_names_, _order_)
                        )
        #
        return enum_class

    def __bool__(cls):
        """
        classes/types should always be True.
        """
        return True

    def __call__(cls, value, names=_not_given, *values, module=None, qualname=None, type=None, start=1, boundary=None):
        """
        Either returns an existing member, or creates a new enum class.

        This method is used both when an enum class is given a value to
        match to an enumeration member (i.e. Color(3)) and for the
        functional API (i.e. Color = Enum('Color', names='RED GREEN BLUE')).

        The value lookup branch is chosen if the enum is final.

        When used for the functional API:

        `value` will be the name of the new class.

        `names` should be either a string of white-space/comma delimited
        names (values will start at `start`), or an iterator/mapping of
        name, value pairs.

        `module` should be set to the module this class is being created in;
        if it is not set, an attempt to find that module will be made, but
        if it fails the class will not be picklable.

        `qualname` should be set to the actual location this class can be
        found at in its module; by default it is set to the global scope.
        If this is not correct, unpickling will fail in some circumstances.

        `type`, if set, will be mixed in as the first base class.
        """
        if cls._member_map_:
            # simple value lookup if members exist
            if names is not _not_given:
                value = (value, names) + values
            return cls.__new__(cls, value)
        # otherwise, functional API: we're creating a new Enum type
        if names is _not_given and type is None:
            # no body? no data-type? possibly wrong usage
            raise TypeError(
                    f"{cls} has no members; specify `names=()` if you meant to create a new, empty, enum"
                    )
        return cls._create_(
                class_name=value,
                names=None if names is _not_given else names,
                module=module,
                qualname=qualname,
                type=type,
                start=start,
                boundary=boundary,
                )

    def __contains__(cls, value):
        """Return True if `value` is in `cls`.

        `value` is in `cls` if:
        1) `value` is a member of `cls`, or
        2) `value` is the value of one of the `cls`'s members.
        3) `value` is a pseudo-member (flags)
        """
        if isinstance(value, cls):
            return True
        if issubclass(cls, Flag):
            try:
                result = cls._missing_(value)
                return isinstance(result, cls)
            except ValueError:
                pass
        return (
                value in cls._unhashable_values_    # both structures are lists
                or value in cls._hashable_values_
                )

    def __delattr__(cls, attr):
        # nicer error message when someone tries to delete an attribute
        # (see issue19025).
        if attr in cls._member_map_:
            raise AttributeError("%r cannot delete member %r." % (cls.__name__, attr))
        super().__delattr__(attr)

    def __dir__(cls):
        interesting = set([
                '__class__', '__contains__', '__doc__', '__getitem__',
                '__iter__', '__len__', '__members__', '__module__',
                '__name__', '__qualname__',
                ]
                + cls._member_names_
                )
        if cls._new_member_ is not object.__new__:
            interesting.add('__new__')
        if cls.__init_subclass__ is not object.__init_subclass__:
            interesting.add('__init_subclass__')
        if cls._member_type_ is object:
            return sorted(interesting)
        else:
            # return whatever mixed-in data type has
            return sorted(set(dir(cls._member_type_)) | interesting)

    def __getitem__(cls, name):
        """
        Return the member matching `name`.
        """
        return cls._member_map_[name]

    def __iter__(cls):
        """
        Return members in definition order.
        """
        return (cls._member_map_[name] for name in cls._member_names_)

    def __len__(cls):
        """
        Return the number of members (no aliases)
        """
        return len(cls._member_names_)

    @bltns.property
    def __members__(cls):
        """
        Returns a mapping of member name->value.

        This mapping lists all enum members, including aliases.  Note that
        this is a read-only view of the internal mapping.
        """
        return MappingProxyType(cls._member_map_)

    def __repr__(cls):
        if Flag is not None and issubclass(cls, Flag):
            return "<flag %r>" % cls.__name__
        else:
            return "<enum %r>" % cls.__name__

    def __reversed__(cls):
        """
        Return members in reverse definition order.
        """
        return (cls._member_map_[name] for name in reversed(cls._member_names_))

    def __setattr__(cls, name, value):
        """
        Block attempts to reassign Enum members.

        A simple assignment to the class namespace only changes one of the
        several possible ways to get an Enum member from the Enum class,
        resulting in an inconsistent Enumeration.
        """
        member_map = cls.__dict__.get('_member_map_', {})
        if name in member_map:
            raise AttributeError('cannot reassign member %r' % (name, ))
        super().__setattr__(name, value)

    def _create_(cls, class_name, names, *, module=None, qualname=None, type=None, start=1, boundary=None):
        """
        Convenience method to create a new Enum class.

        `names` can be:

        * A string containing member names, separated either with spaces or
          commas.  Values are incremented by 1 from `start`.
        * An iterable of member names.  Values are incremented by 1 from `start`.
        * An iterable of (member name, value) pairs.
        * A mapping of member name -> value pairs.
        """
        metacls = cls.__class__
        bases = (cls, ) if type is None else (type, cls)
        _, first_enum = cls._get_mixins_(class_name, bases)
        classdict = metacls.__prepare__(class_name, bases)

        # special processing needed for names?
        if isinstance(names, str):
            names = names.replace(',', ' ').split()
        if isinstance(names, (tuple, list)) and names and isinstance(names[0], str):
            original_names, names = names, []
            last_values = []
            for count, name in enumerate(original_names):
                value = first_enum._generate_next_value_(name, start, count, last_values[:])
                last_values.append(value)
                names.append((name, value))
        if names is None:
            names = ()

        # Here, names is either an iterable of (name, value) or a mapping.
        for item in names:
            if isinstance(item, str):
                member_name, member_value = item, names[item]
            else:
                member_name, member_value = item
            classdict[member_name] = member_value

        if module is None:
            try:
                module = sys._getframemodulename(2)
            except AttributeError:
                # Fall back on _getframe if _getframemodulename is missing
                try:
                    module = sys._getframe(2).f_globals['__name__']
                except (AttributeError, ValueError, KeyError):
                    pass
        if module is None:
            _make_class_unpicklable(classdict)
        else:
            classdict['__module__'] = module
        if qualname is not None:
            classdict['__qualname__'] = qualname

        return metacls.__new__(metacls, class_name, bases, classdict, boundary=boundary)

    def _convert_(cls, name, module, filter, source=None, *, boundary=None, as_global=False):
        """
        Create a new Enum subclass that replaces a collection of global constants
        """
        # convert all constants from source (or module) that pass filter() to
        # a new Enum called name, and export the enum and its members back to
        # module;
        # also, replace the __reduce_ex__ method so unpickling works in
        # previous Python versions
        module_globals = sys.modules[module].__dict__
        if source:
            source = source.__dict__
        else:
            source = module_globals
        # _value2member_map_ is populated in the same order every time
        # for a consistent reverse mapping of number to name when there
        # are multiple names for the same number.
        members = [
                (name, value)
                for name, value in source.items()
                if filter(name)]
        try:
            # sort by value
            members.sort(key=lambda t: (t[1], t[0]))
        except TypeError:
            # unless some values aren't comparable, in which case sort by name
            members.sort(key=lambda t: t[0])
        body = {t[0]: t[1] for t in members}
        body['__module__'] = module
        tmp_cls = type(name, (object, ), body)
        cls = _simple_enum(etype=cls, boundary=boundary or KEEP)(tmp_cls)
        if as_global:
            global_enum(cls)
        else:
            sys.modules[cls.__module__].__dict__.update(cls.__members__)
        module_globals[name] = cls
        return cls

    @classmethod
    def _check_for_existing_members_(mcls, class_name, bases):
        for chain in bases:
            for base in chain.__mro__:
                if isinstance(base, EnumType) and base._member_names_:
                    raise TypeError(
                            "<enum %r> cannot extend %r"
                            % (class_name, base)
                            )

    @classmethod
    def _get_mixins_(mcls, class_name, bases):
        """
        Returns the type for creating enum members, and the first inherited
        enum class.

        bases: the tuple of bases that was given to __new__
        """
        if not bases:
            return object, Enum
        # ensure final parent class is an Enum derivative, find any concrete
        # data type, and check that Enum has no members
        first_enum = bases[-1]
        if not isinstance(first_enum, EnumType):
            raise TypeError("new enumerations should be created as "
                    "`EnumName([mixin_type, ...] [data_type,] enum_type)`")
        member_type = mcls._find_data_type_(class_name, bases) or object
        return member_type, first_enum

    @classmethod
    def _find_data_repr_(mcls, class_name, bases):
        for chain in bases:
            for base in chain.__mro__:
                if base is object:
                    continue
                elif isinstance(base, EnumType):
                    # if we hit an Enum, use it's _value_repr_
                    return base._value_repr_
                elif '__repr__' in base.__dict__:
                    # this is our data repr
                    # double-check if a dataclass with a default __repr__
                    if (
                            '__dataclass_fields__' in base.__dict__
                            and '__dataclass_params__' in base.__dict__
                            and base.__dict__['__dataclass_params__'].repr
                        ):
                        return _dataclass_repr
                    else:
                        return base.__dict__['__repr__']
        return None

    @classmethod
    def _find_data_type_(mcls, class_name, bases):
        # a datatype has a __new__ method, or a __dataclass_fields__ attribute
        data_types = set()
        base_chain = set()
        for chain in bases:
            candidate = None
            for base in chain.__mro__:
                base_chain.add(base)
                if base is object:
                    continue
                elif isinstance(base, EnumType):
                    if base._member_type_ is not object:
                        data_types.add(base._member_type_)
                        break
                elif '__new__' in base.__dict__ or '__dataclass_fields__' in base.__dict__:
                    data_types.add(candidate or base)
                    break
                else:
                    candidate = candidate or base
        if len(data_types) > 1:
            raise TypeError('too many data types for %r: %r' % (class_name, data_types))
        elif data_types:
            return data_types.pop()
        else:
            return None

    @classmethod
    def _find_new_(mcls, classdict, member_type, first_enum):
        """
        Returns the __new__ to be used for creating the enum members.

        classdict: the class dictionary given to __new__
        member_type: the data type whose __new__ will be used by default
        first_enum: enumeration to check for an overriding __new__
        """
        # now find the correct __new__, checking to see of one was defined
        # by the user; also check earlier enum classes in case a __new__ was
        # saved as __new_member__
        __new__ = classdict.get('__new__', None)

        # should __new__ be saved as __new_member__ later?
        save_new = first_enum is not None and __new__ is not None

        if __new__ is None:
            # check all possibles for __new_member__ before falling back to
            # __new__
            for method in ('__new_member__', '__new__'):
                for possible in (member_type, first_enum):
                    target = getattr(possible, method, None)
                    if target not in {
                            None,
                            None.__new__,
                            object.__new__,
                            Enum.__new__,
                            }:
                        __new__ = target
                        break
                if __new__ is not None:
                    break
            else:
                __new__ = object.__new__

        # if a non-object.__new__ is used then whatever value/tuple was
        # assigned to the enum member name will be passed to __new__ and to the
        # new enum member's __init__
        if first_enum is None or __new__ in (Enum.__new__, object.__new__):
            use_args = False
        else:
            use_args = True
        return __new__, save_new, use_args

    def _add_member_(cls, name, member):
        # _value_ structures are not updated
        if name in cls._member_map_:
            if cls._member_map_[name] is not member:
                raise NameError('%r is already bound: %r' % (name, cls._member_map_[name]))
            return
        #
        # if necessary, get redirect in place and then add it to _member_map_
        found_descriptor = None
        descriptor_type = None
        class_type = None
        for base in cls.__mro__[1:]:
            attr = base.__dict__.get(name)
            if attr is not None:
                if isinstance(attr, (property, DynamicClassAttribute)):
                    found_descriptor = attr
                    class_type = base
                    descriptor_type = 'enum'
                    break
                elif _is_descriptor(attr):
                    found_descriptor = attr
                    descriptor_type = descriptor_type or 'desc'
                    class_type = class_type or base
                    continue
                else:
                    descriptor_type = 'attr'
                    class_type = base
        if found_descriptor:
            redirect = property()
            redirect.member = member
            redirect.__set_name__(cls, name)
            if descriptor_type in ('enum', 'desc'):
                # earlier descriptor found; copy fget, fset, fdel to this one.
                redirect.fget = getattr(found_descriptor, 'fget', None)
                redirect._get = getattr(found_descriptor, '__get__', None)
                redirect.fset = getattr(found_descriptor, 'fset', None)
                redirect._set = getattr(found_descriptor, '__set__', None)
                redirect.fdel = getattr(found_descriptor, 'fdel', None)
                redirect._del = getattr(found_descriptor, '__delete__', None)
            redirect._attr_type = descriptor_type
            redirect._cls_type = class_type
            setattr(cls, name, redirect)
        else:
            setattr(cls, name, member)
        # now add to _member_map_ (even aliases)
        cls._member_map_[name] = member

EnumMeta = EnumType         # keep EnumMeta name for backwards compatibility


class Enum(metaclass=EnumType):
    """
    Create a collection of name/value pairs.

    Example enumeration:

    >>> class Color(Enum):
    ...     RED = 1
    ...     BLUE = 2
    ...     GREEN = 3

    Access them by:

    - attribute access:

      >>> Color.RED
      <Color.RED: 1>

    - value lookup:

      >>> Color(1)
      <Color.RED: 1>

    - name lookup:

      >>> Color['RED']
      <Color.RED: 1>

    Enumerations can be iterated over, and know how many members they have:

    >>> len(Color)
    3

    >>> list(Color)
    [<Color.RED: 1>, <Color.BLUE: 2>, <Color.GREEN: 3>]

    Methods can be added to enumerations, and members can have their own
    attributes -- see the documentation for details.
    """

    @classmethod
    def __signature__(cls):
        if cls._member_names_:
            return '(*values)'
        else:
            return '(new_class_name, /, names, *, module=None, qualname=None, type=None, start=1, boundary=None)'

    def __new__(cls, value):
        # all enum instances are actually created during class construction
        # without calling this method; this method is called by the metaclass'
        # __call__ (i.e. Color(3) ), and by pickle
        if type(value) is cls:
            # For lookups like Color(Color.RED)
            return value
        # by-value search for a matching enum member
        # see if it's in the reverse mapping (for hashable values)
        try:
            return cls._value2member_map_[value]
        except KeyError:
            # Not found, no need to do long O(n) search
            pass
        except TypeError:
            # not there, now do long search -- O(n) behavior
            for name, unhashable_values in cls._unhashable_values_map_.items():
                if value in unhashable_values:
                    return cls[name]
            for name, member in cls._member_map_.items():
                if value == member._value_:
                    return cls[name]
        # still not found -- verify that members exist, in-case somebody got here mistakenly
        # (such as via super when trying to override __new__)
        if not cls._member_map_:
            if getattr(cls, '_%s__in_progress' % cls.__name__, False):
                raise TypeError('do not use `super().__new__; call the appropriate __new__ directly') from None
            raise TypeError("%r has no members defined" % cls)
        #
        # still not found -- try _missing_ hook
        try:
            exc = None
            result = cls._missing_(value)
        except Exception as e:
            exc = e
            result = None
        try:
            if isinstance(result, cls):
                return result
            elif (
                    Flag is not None and issubclass(cls, Flag)
                    and cls._boundary_ is EJECT and isinstance(result, int)
                ):
                return result
            else:
                ve_exc = ValueError("%r is not a valid %s" % (value, cls.__qualname__))
                if result is None and exc is None:
                    raise ve_exc
                elif exc is None:
                    exc = TypeError(
                            'error in %s._missing_: returned %r instead of None or a valid member'
                            % (cls.__name__, result)
                            )
                if not isinstance(exc, ValueError):
                    exc.__context__ = ve_exc
                raise exc
        finally:
            # ensure all variables that could hold an exception are destroyed
            exc = None
            ve_exc = None

    def __init__(self, *args, **kwds):
        pass

    def _add_alias_(self, name):
        self.__class__._add_member_(name, self)

    def _add_value_alias_(self, value):
        cls = self.__class__
        try:
            if value in cls._value2member_map_:
                if cls._value2member_map_[value] is not self:
                    raise ValueError('%r is already bound: %r' % (value, cls._value2member_map_[value]))
                return
        except TypeError:
            # unhashable value, do long search
            for m in cls._member_map_.values():
                if m._value_ == value:
                    if m is not self:
                        raise ValueError('%r is already bound: %r' % (value, cls._value2member_map_[value]))
                    return
        try:
            # This may fail if value is not hashable. We can't add the value
            # to the map, and by-value lookups for this value will be
            # linear.
            cls._value2member_map_.setdefault(value, self)
            cls._hashable_values_.append(value)
        except TypeError:
            # keep track of the value in a list so containment checks are quick
            cls._unhashable_values_.append(value)
            cls._unhashable_values_map_.setdefault(self.name, []).append(value)

    @staticmethod
    def _generate_next_value_(name, start, count, last_values):
        """
        Generate the next value when not given.

        name: the name of the member
        start: the initial start value or None
        count: the number of existing members
        last_values: the list of values assigned
        """
        if not last_values:
            return start
        try:
            last_value = sorted(last_values).pop()
        except TypeError:
            raise TypeError('unable to sort non-numeric values') from None
        try:
            return last_value + 1
        except TypeError:
            raise TypeError('unable to increment %r' % (last_value, )) from None

    @classmethod
    def _missing_(cls, value):
        return None

    def __repr__(self):
        v_repr = self.__class__._value_repr_ or repr
        return "<%s.%s: %s>" % (self.__class__.__name__, self._name_, v_repr(self._value_))

    def __str__(self):
        return "%s.%s" % (self.__class__.__name__, self._name_, )

    def __dir__(self):
        """
        Returns public methods and other interesting attributes.
        """
        interesting = set()
        if self.__class__._member_type_ is not object:
            interesting = set(object.__dir__(self))
        for name in getattr(self, '__dict__', []):
            if name[0] != '_' and name not in self._member_map_:
                interesting.add(name)
        for cls in self.__class__.mro():
            for name, obj in cls.__dict__.items():
                if name[0] == '_':
                    continue
                if isinstance(obj, property):
                    # that's an enum.property
                    if obj.fget is not None or name not in self._member_map_:
                        interesting.add(name)
                    else:
                        # in case it was added by `dir(self)`
                        interesting.discard(name)
                elif name not in self._member_map_:
                    interesting.add(name)
        names = sorted(
                set(['__class__', '__doc__', '__eq__', '__hash__', '__module__'])
                | interesting
                )
        return names

    def __format__(self, format_spec):
        return str.__format__(str(self), format_spec)

    def __hash__(self):
        return hash(self._name_)

    def __reduce_ex__(self, proto):
        return self.__class__, (self._value_, )

    def __deepcopy__(self,memo):
        return self

    def __copy__(self):
        return self

    # enum.property is used to provide access to the `name` and
    # `value` attributes of enum members while keeping some measure of
    # protection from modification, while still allowing for an enumeration
    # to have members named `name` and `value`.  This works because each
    # instance of enum.property saves its companion member, which it returns
    # on class lookup; on instance lookup it either executes a provided function
    # or raises an AttributeError.

    @property
    def name(self):
        """The name of the Enum member."""
        return self._name_

    @property
    def value(self):
        """The value of the Enum member."""
        return self._value_


class ReprEnum(Enum):
    """
    Only changes the repr(), leaving str() and format() to the mixed-in type.
    """


class IntEnum(int, ReprEnum):
    """
    Enum where members are also (and must be) ints
    """


class StrEnum(str, ReprEnum):
    """
    Enum where members are also (and must be) strings
    """

    def __new__(cls, *values):
        "values must already be of type `str`"
        if len(values) > 3:
            raise TypeError('too many arguments for str(): %r' % (values, ))
        if len(values) == 1:
            # it must be a string
            if not isinstance(values[0], str):
                raise TypeError('%r is not a string' % (values[0], ))
        if len(values) >= 2:
            # check that encoding argument is a string
            if not isinstance(values[1], str):
                raise TypeError('encoding must be a string, not %r' % (values[1], ))
        if len(values) == 3:
            # check that errors argument is a string
            if not isinstance(values[2], str):
                raise TypeError('errors must be a string, not %r' % (values[2]))
        value = str(*values)
        member = str.__new__(cls, value)
        member._value_ = value
        return member

    @staticmethod
    def _generate_next_value_(name, start, count, last_values):
        """
        Return the lower-cased version of the member name.
        """
        return name.lower()


def pickle_by_global_name(self, proto):
    # should not be used with Flag-type enums
    return self.name
_reduce_ex_by_global_name = pickle_by_global_name

def pickle_by_enum_name(self, proto):
    # should not be used with Flag-type enums
    return getattr, (self.__class__, self._name_)

class FlagBoundary(StrEnum):
    """
    control how out of range values are handled
    "strict" -> error is raised             [default for Flag]
    "conform" -> extra bits are discarded
    "eject" -> lose flag status
    "keep" -> keep flag status and all bits [default for IntFlag]
    """
    STRICT = auto()
    CONFORM = auto()
    EJECT = auto()
    KEEP = auto()
STRICT, CONFORM, EJECT, KEEP = FlagBoundary


class Flag(Enum, boundary=STRICT):
    """
    Support for flags
    """

    _numeric_repr_ = repr

    @staticmethod
    def _generate_next_value_(name, start, count, last_values):
        """
        Generate the next value when not given.

        name: the name of the member
        start: the initial start value or None
        count: the number of existing members
        last_values: the last value assigned or None
        """
        if not count:
            return start if start is not None else 1
        last_value = max(last_values)
        try:
            high_bit = _high_bit(last_value)
        except Exception:
            raise TypeError('invalid flag value %r' % last_value) from None
        return 2 ** (high_bit+1)

    @classmethod
    def _iter_member_by_value_(cls, value):
        """
        Extract all members from the value in definition (i.e. increasing value) order.
        """
        for val in _iter_bits_lsb(value & cls._flag_mask_):
            yield cls._value2member_map_.get(val)

    _iter_member_ = _iter_member_by_value_

    @classmethod
    def _iter_member_by_def_(cls, value):
        """
        Extract all members from the value in definition order.
        """
        yield from sorted(
                cls._iter_member_by_value_(value),
                key=lambda m: m._sort_order_,
                )

    @classmethod
    def _missing_(cls, value):
        """
        Create a composite member containing all canonical members present in `value`.

        If non-member values are present, result depends on `_boundary_` setting.
        """
        if not isinstance(value, int):
            raise ValueError(
                    "%r is not a valid %s" % (value, cls.__qualname__)
                    )
        # check boundaries
        # - value must be in range (e.g. -16 <-> +15, i.e. ~15 <-> 15)
        # - value must not include any skipped flags (e.g. if bit 2 is not
        #   defined, then 0d10 is invalid)
        flag_mask = cls._flag_mask_
        singles_mask = cls._singles_mask_
        all_bits = cls._all_bits_
        neg_value = None
        if (
                not ~all_bits <= value <= all_bits
                or value & (all_bits ^ flag_mask)
            ):
            if cls._boundary_ is STRICT:
                max_bits = max(value.bit_length(), flag_mask.bit_length())
                raise ValueError(
                        "%r invalid value %r\n    given %s\n  allowed %s" % (
                            cls, value, bin(value, max_bits), bin(flag_mask, max_bits),
                            ))
            elif cls._boundary_ is CONFORM:
                value = value & flag_mask
            elif cls._boundary_ is EJECT:
                return value
            elif cls._boundary_ is KEEP:
                if value < 0:
                    value = (
                            max(all_bits+1, 2**(value.bit_length()))
                            + value
                            )
            else:
                raise ValueError(
                        '%r unknown flag boundary %r' % (cls, cls._boundary_, )
                        )
        if value < 0:
            neg_value = value
            value = all_bits + 1 + value
        # get members and unknown
        unknown = value & ~flag_mask
        aliases = value & ~singles_mask
        member_value = value & singles_mask
        if unknown and cls._boundary_ is not KEEP:
            raise ValueError(
                    '%s(%r) -->  unknown values %r [%s]'
                    % (cls.__name__, value, unknown, bin(unknown))
                    )
        # normal Flag?
        if cls._member_type_ is object:
            # construct a singleton enum pseudo-member
            pseudo_member = object.__new__(cls)
        else:
            pseudo_member = cls._member_type_.__new__(cls, value)
        if not hasattr(pseudo_member, '_value_'):
            pseudo_member._value_ = value
        if member_value or aliases:
            members = []
            combined_value = 0
            for m in cls._iter_member_(member_value):
                members.append(m)
                combined_value |= m._value_
            if aliases:
                value = member_value | aliases
                for n, pm in cls._member_map_.items():
                    if pm not in members and pm._value_ and pm._value_ & value == pm._value_:
                        members.append(pm)
                        combined_value |= pm._value_
            unknown = value ^ combined_value
            pseudo_member._name_ = '|'.join([m._name_ for m in members])
            if not combined_value:
                pseudo_member._name_ = None
            elif unknown and cls._boundary_ is STRICT:
                raise ValueError('%r: no members with value %r' % (cls, unknown))
            elif unknown:
                pseudo_member._name_ += '|%s' % cls._numeric_repr_(unknown)
        else:
            pseudo_member._name_ = None
        # use setdefault in case another thread already created a composite
        # with this value
        # note: zero is a special case -- always add it
        pseudo_member = cls._value2member_map_.setdefault(value, pseudo_member)
        if neg_value is not None:
            cls._value2member_map_[neg_value] = pseudo_member
        return pseudo_member

    def __contains__(self, other):
        """
        Returns True if self has at least the same flags set as other.
        """
        if not isinstance(other, self.__class__):
            raise TypeError(
                "unsupported operand type(s) for 'in': %r and %r" % (
                    type(other).__qualname__, self.__class__.__qualname__))
        return other._value_ & self._value_ == other._value_

    def __iter__(self):
        """
        Returns flags in definition order.
        """
        yield from self._iter_member_(self._value_)

    def __len__(self):
        return self._value_.bit_count()

    def __repr__(self):
        cls_name = self.__class__.__name__
        v_repr = self.__class__._value_repr_ or repr
        if self._name_ is None:
            return "<%s: %s>" % (cls_name, v_repr(self._value_))
        else:
            return "<%s.%s: %s>" % (cls_name, self._name_, v_repr(self._value_))

    def __str__(self):
        cls_name = self.__class__.__name__
        if self._name_ is None:
            return '%s(%r)' % (cls_name, self._value_)
        else:
            return "%s.%s" % (cls_name, self._name_)

    def __bool__(self):
        return bool(self._value_)

    def _get_value(self, flag):
        if isinstance(flag, self.__class__):
            return flag._value_
        elif self._member_type_ is not object and isinstance(flag, self._member_type_):
            return flag
        return NotImplemented

    def __or__(self, other):
        other_value = self._get_value(other)
        if other_value is NotImplemented:
            return NotImplemented

        for flag in self, other:
            if self._get_value(flag) is None:
                raise TypeError(f"'{flag}' cannot be combined with other flags with |")
        value = self._value_
        return self.__class__(value | other_value)

    def __and__(self, other):
        other_value = self._get_value(other)
        if other_value is NotImplemented:
            return NotImplemented

        for flag in self, other:
            if self._get_value(flag) is None:
                raise TypeError(f"'{flag}' cannot be combined with other flags with &")
        value = self._value_
        return self.__class__(value & other_value)

    def __xor__(self, other):
        other_value = self._get_value(other)
        if other_value is NotImplemented:
            return NotImplemented

        for flag in self, other:
            if self._get_value(flag) is None:
                raise TypeError(f"'{flag}' cannot be combined with other flags with ^")
        value = self._value_
        return self.__class__(value ^ other_value)

    def __invert__(self):
        if self._get_value(self) is None:
            raise TypeError(f"'{self}' cannot be inverted")

        if self._inverted_ is None:
            if self._boundary_ in (EJECT, KEEP):
                self._inverted_ = self.__class__(~self._value_)
            else:
                self._inverted_ = self.__class__(self._singles_mask_ & ~self._value_)
        return self._inverted_

    __rand__ = __and__
    __ror__ = __or__
    __rxor__ = __xor__


class IntFlag(int, ReprEnum, Flag, boundary=KEEP):
    """
    Support for integer-based Flags
    """


def _high_bit(value):
    """
    returns index of highest bit, or -1 if value is zero or negative
    """
    return value.bit_length() - 1

def unique(enumeration):
    """
    Class decorator for enumerations ensuring unique member values.
    """
    duplicates = []
    for name, member in enumeration.__members__.items():
        if name != member.name:
            duplicates.append((name, member.name))
    if duplicates:
        alias_details = ', '.join(
                ["%s -> %s" % (alias, name) for (alias, name) in duplicates])
        raise ValueError('duplicate values found in %r: %s' %
                (enumeration, alias_details))
    return enumeration

def _dataclass_repr(self):
    dcf = self.__dataclass_fields__
    return ', '.join(
            '%s=%r' % (k, getattr(self, k))
            for k in dcf.keys()
            if dcf[k].repr
            )

def global_enum_repr(self):
    """
    use module.enum_name instead of class.enum_name

    the module is the last module in case of a multi-module name
    """
    module = self.__class__.__module__.split('.')[-1]
    return '%s.%s' % (module, self._name_)

def global_flag_repr(self):
    """
    use module.flag_name instead of class.flag_name

    the module is the last module in case of a multi-module name
    """
    module = self.__class__.__module__.split('.')[-1]
    cls_name = self.__class__.__name__
    if self._name_ is None:
        return "%s.%s(%r)" % (module, cls_name, self._value_)
    if _is_single_bit(self._value_):
        return '%s.%s' % (module, self._name_)
    if self._boundary_ is not FlagBoundary.KEEP:
        return '|'.join(['%s.%s' % (module, name) for name in self.name.split('|')])
    else:
        name = []
        for n in self._name_.split('|'):
            if n[0].isdigit():
                name.append(n)
            else:
                name.append('%s.%s' % (module, n))
        return '|'.join(name)

def global_str(self):
    """
    use enum_name instead of class.enum_name
    """
    if self._name_ is None:
        cls_name = self.__class__.__name__
        return "%s(%r)" % (cls_name, self._value_)
    else:
        return self._name_

def global_enum(cls, update_str=False):
    """
    decorator that makes the repr() of an enum member reference its module
    instead of its class; also exports all members to the enum's module's
    global namespace
    """
    if issubclass(cls, Flag):
        cls.__repr__ = global_flag_repr
    else:
        cls.__repr__ = global_enum_repr
    if not issubclass(cls, ReprEnum) or update_str:
        cls.__str__ = global_str
    sys.modules[cls.__module__].__dict__.update(cls.__members__)
    return cls

def _simple_enum(etype=Enum, *, boundary=None, use_args=None):
    """
    Class decorator that converts a normal class into an :class:`Enum`.  No
    safety checks are done, and some advanced behavior (such as
    :func:`__init_subclass__`) is not available.  Enum creation can be faster
    using :func:`_simple_enum`.

        >>> from enum import Enum, _simple_enum
        >>> @_simple_enum(Enum)
        ... class Color:
        ...     RED = auto()
        ...     GREEN = auto()
        ...     BLUE = auto()
        >>> Color
        <enum 'Color'>
    """
    def convert_class(cls):
        nonlocal use_args
        cls_name = cls.__name__
        if use_args is None:
            use_args = etype._use_args_
        __new__ = cls.__dict__.get('__new__')
        if __new__ is not None:
            new_member = __new__.__func__
        else:
            new_member = etype._member_type_.__new__
        attrs = {}
        body = {}
        if __new__ is not None:
            body['__new_member__'] = new_member
        body['_new_member_'] = new_member
        body['_use_args_'] = use_args
        body['_generate_next_value_'] = gnv = etype._generate_next_value_
        body['_member_names_'] = member_names = []
        body['_member_map_'] = member_map = {}
        body['_value2member_map_'] = value2member_map = {}
        body['_hashable_values_'] = hashable_values = []
        body['_unhashable_values_'] = unhashable_values = []
        body['_unhashable_values_map_'] = {}
        body['_member_type_'] = member_type = etype._member_type_
        body['_value_repr_'] = etype._value_repr_
        if issubclass(etype, Flag):
            body['_boundary_'] = boundary or etype._boundary_
            body['_flag_mask_'] = None
            body['_all_bits_'] = None
            body['_singles_mask_'] = None
            body['_inverted_'] = None
            body['__or__'] = Flag.__or__
            body['__xor__'] = Flag.__xor__
            body['__and__'] = Flag.__and__
            body['__ror__'] = Flag.__ror__
            body['__rxor__'] = Flag.__rxor__
            body['__rand__'] = Flag.__rand__
            body['__invert__'] = Flag.__invert__
        for name, obj in cls.__dict__.items():
            if name in ('__dict__', '__weakref__'):
                continue
            if _is_dunder(name) or _is_private(cls_name, name) or _is_sunder(name) or _is_descriptor(obj):
                body[name] = obj
            else:
                attrs[name] = obj
        if cls.__dict__.get('__doc__') is None:
            body['__doc__'] = 'An enumeration.'
        #
        # double check that repr and friends are not the mixin's or various
        # things break (such as pickle)
        # however, if the method is defined in the Enum itself, don't replace
        # it
        enum_class = type(cls_name, (etype, ), body, boundary=boundary, _simple=True)
        for name in ('__repr__', '__str__', '__format__', '__reduce_ex__'):
            if name not in body:
                # check for mixin overrides before replacing
                enum_method = getattr(etype, name)
                found_method = getattr(enum_class, name)
                object_method = getattr(object, name)
                data_type_method = getattr(member_type, name)
                if found_method in (data_type_method, object_method):
                    setattr(enum_class, name, enum_method)
        gnv_last_values = []
        if issubclass(enum_class, Flag):
            # Flag / IntFlag
            single_bits = multi_bits = 0
            for name, value in attrs.items():
                if isinstance(value, auto) and auto.value is _auto_null:
                    value = gnv(name, 1, len(member_names), gnv_last_values)
                # create basic member (possibly isolate value for alias check)
                if use_args:
                    if not isinstance(value, tuple):
                        value = (value, )
                    member = new_member(enum_class, *value)
                    value = value[0]
                else:
                    member = new_member(enum_class)
                if __new__ is None:
                    member._value_ = value
                # now check if alias
                try:
                    contained = value2member_map.get(member._value_)
                except TypeError:
                    contained = None
                    if member._value_ in unhashable_values or member.value in hashable_values:
                        for m in enum_class:
                            if m._value_ == member._value_:
                                contained = m
                                break
                if contained is not None:
                    # an alias to an existing member
                    contained._add_alias_(name)
                else:
                    # finish creating member
                    member._name_ = name
                    member.__objclass__ = enum_class
                    member.__init__(value)
                    member._sort_order_ = len(member_names)
                    if name not in ('name', 'value'):
                        setattr(enum_class, name, member)
                        member_map[name] = member
                    else:
                        enum_class._add_member_(name, member)
                    value2member_map[value] = member
                    hashable_values.append(value)
                    if _is_single_bit(value):
                        # not a multi-bit alias, record in _member_names_ and _flag_mask_
                        member_names.append(name)
                        single_bits |= value
                    else:
                        multi_bits |= value
                    gnv_last_values.append(value)
            enum_class._flag_mask_ = single_bits | multi_bits
            enum_class._singles_mask_ = single_bits
            enum_class._all_bits_ = 2 ** ((single_bits|multi_bits).bit_length()) - 1
            # set correct __iter__
            member_list = [m._value_ for m in enum_class]
            if member_list != sorted(member_list):
                enum_class._iter_member_ = enum_class._iter_member_by_def_
        else:
            # Enum / IntEnum / StrEnum
            for name, value in attrs.items():
                if isinstance(value, auto):
                    if value.value is _auto_null:
                        value.value = gnv(name, 1, len(member_names), gnv_last_values)
                    value = value.value
                # create basic member (possibly isolate value for alias check)
                if use_args:
                    if not isinstance(value, tuple):
                        value = (value, )
                    member = new_member(enum_class, *value)
                    value = value[0]
                else:
                    member = new_member(enum_class)
                if __new__ is None:
                    member._value_ = value
                # now check if alias
                try:
                    contained = value2member_map.get(member._value_)
                except TypeError:
                    contained = None
                    if member._value_ in unhashable_values or member._value_ in hashable_values:
                        for m in enum_class:
                            if m._value_ == member._value_:
                                contained = m
                                break
                if contained is not None:
                    # an alias to an existing member
                    contained._add_alias_(name)
                else:
                    # finish creating member
                    member._name_ = name
                    member.__objclass__ = enum_class
                    member.__init__(value)
                    member._sort_order_ = len(member_names)
                    if name not in ('name', 'value'):
                        setattr(enum_class, name, member)
                        member_map[name] = member
                    else:
                        enum_class._add_member_(name, member)
                    member_names.append(name)
                    gnv_last_values.append(value)
                    try:
                        # This may fail if value is not hashable. We can't add the value
                        # to the map, and by-value lookups for this value will be
                        # linear.
                        enum_class._value2member_map_.setdefault(value, member)
                        if value not in hashable_values:
                            hashable_values.append(value)
                    except TypeError:
                        # keep track of the value in a list so containment checks are quick
                        enum_class._unhashable_values_.append(value)
                        enum_class._unhashable_values_map_.setdefault(name, []).append(value)
        if '__new__' in body:
            enum_class.__new_member__ = enum_class.__new__
        enum_class.__new__ = Enum.__new__
        return enum_class
    return convert_class

@_simple_enum(StrEnum)
class EnumCheck:
    """
    various conditions to check an enumeration for
    """
    CONTINUOUS = "no skipped integer values"
    NAMED_FLAGS = "multi-flag aliases may not contain unnamed flags"
    UNIQUE = "one name per value"
CONTINUOUS, NAMED_FLAGS, UNIQUE = EnumCheck


class verify:
    """
    Check an enumeration for various constraints. (see EnumCheck)
    """
    def __init__(self, *checks):
        self.checks = checks
    def __call__(self, enumeration):
        checks = self.checks
        cls_name = enumeration.__name__
        if Flag is not None and issubclass(enumeration, Flag):
            enum_type = 'flag'
        elif issubclass(enumeration, Enum):
            enum_type = 'enum'
        else:
            raise TypeError("the 'verify' decorator only works with Enum and Flag")
        for check in checks:
            if check is UNIQUE:
                # check for duplicate names
                duplicates = []
                for name, member in enumeration.__members__.items():
                    if name != member.name:
                        duplicates.append((name, member.name))
                if duplicates:
                    alias_details = ', '.join(
                            ["%s -> %s" % (alias, name) for (alias, name) in duplicates])
                    raise ValueError('aliases found in %r: %s' %
                            (enumeration, alias_details))
            elif check is CONTINUOUS:
                values = set(e.value for e in enumeration)
                if len(values) < 2:
                    continue
                low, high = min(values), max(values)
                missing = []
                if enum_type == 'flag':
                    # check for powers of two
                    for i in range(_high_bit(low)+1, _high_bit(high)):
                        if 2**i not in values:
                            missing.append(2**i)
                elif enum_type == 'enum':
                    # check for missing consecutive integers
                    for i in range(low+1, high):
                        if i not in values:
                            missing.append(i)
                else:
                    raise Exception('verify: unknown type %r' % enum_type)
                if missing:
                    raise ValueError(('invalid %s %r: missing values %s' % (
                            enum_type, cls_name, ', '.join((str(m) for m in missing)))
                            )[:256])
                            # limit max length to protect against DOS attacks
            elif check is NAMED_FLAGS:
                # examine each alias and check for unnamed flags
                member_names = enumeration._member_names_
                member_values = [m.value for m in enumeration]
                missing_names = []
                missing_value = 0
                for name, alias in enumeration._member_map_.items():
                    if name in member_names:
                        # not an alias
                        continue
                    if alias.value < 0:
                        # negative numbers are not checked
                        continue
                    values = list(_iter_bits_lsb(alias.value))
                    missed = [v for v in values if v not in member_values]
                    if missed:
                        missing_names.append(name)
                        for val in missed:
                            missing_value |= val
                if missing_names:
                    if len(missing_names) == 1:
                        alias = 'alias %s is missing' % missing_names[0]
                    else:
                        alias = 'aliases %s and %s are missing' % (
                                ', '.join(missing_names[:-1]), missing_names[-1]
                                )
                    if _is_single_bit(missing_value):
                        value = 'value 0x%x' % missing_value
                    else:
                        value = 'combined values of 0x%x' % missing_value
                    raise ValueError(
                            'invalid Flag %r: %s %s [use enum.show_flag_values(value) for details]'
                            % (cls_name, alias, value)
                            )
        return enumeration

def _test_simple_enum(checked_enum, simple_enum):
    """
    A function that can be used to test an enum created with :func:`_simple_enum`
    against the version created by subclassing :class:`Enum`::

        >>> from enum import Enum, _simple_enum, _test_simple_enum
        >>> @_simple_enum(Enum)
        ... class Color:
        ...     RED = auto()
        ...     GREEN = auto()
        ...     BLUE = auto()
        >>> class CheckedColor(Enum):
        ...     RED = auto()
        ...     GREEN = auto()
        ...     BLUE = auto()
        >>> _test_simple_enum(CheckedColor, Color)

    If differences are found, a :exc:`TypeError` is raised.
    """
    failed = []
    if checked_enum.__dict__ != simple_enum.__dict__:
        checked_dict = checked_enum.__dict__
        checked_keys = list(checked_dict.keys())
        simple_dict = simple_enum.__dict__
        simple_keys = list(simple_dict.keys())
        member_names = set(
                list(checked_enum._member_map_.keys())
                + list(simple_enum._member_map_.keys())
                )
        for key in set(checked_keys + simple_keys):
            if key in ('__module__', '_member_map_', '_value2member_map_', '__doc__',
                       '__static_attributes__', '__firstlineno__'):
                # keys known to be different, or very long
                continue
            elif key in member_names:
                # members are checked below
                continue
            elif key not in simple_keys:
                failed.append("missing key: %r" % (key, ))
            elif key not in checked_keys:
                failed.append("extra key:   %r" % (key, ))
            else:
                checked_value = checked_dict[key]
                simple_value = simple_dict[key]
                if callable(checked_value) or isinstance(checked_value, bltns.property):
                    continue
                if key == '__doc__':
                    # remove all spaces/tabs
                    compressed_checked_value = checked_value.replace(' ','').replace('\t','')
                    compressed_simple_value = simple_value.replace(' ','').replace('\t','')
                    if compressed_checked_value != compressed_simple_value:
                        failed.append("%r:\n         %s\n         %s" % (
                                key,
                                "checked -> %r" % (checked_value, ),
                                "simple  -> %r" % (simple_value, ),
                                ))
                elif checked_value != simple_value:
                    failed.append("%r:\n         %s\n         %s" % (
                            key,
                            "checked -> %r" % (checked_value, ),
                            "simple  -> %r" % (simple_value, ),
                            ))
        failed.sort()
        for name in member_names:
            failed_member = []
            if name not in simple_keys:
                failed.append('missing member from simple enum: %r' % name)
            elif name not in checked_keys:
                failed.append('extra member in simple enum: %r' % name)
            else:
                checked_member_dict = checked_enum[name].__dict__
                checked_member_keys = list(checked_member_dict.keys())
                simple_member_dict = simple_enum[name].__dict__
                simple_member_keys = list(simple_member_dict.keys())
                for key in set(checked_member_keys + simple_member_keys):
                    if key in ('__module__', '__objclass__', '_inverted_'):
                        # keys known to be different or absent
                        continue
                    elif key not in simple_member_keys:
                        failed_member.append("missing key %r not in the simple enum member %r" % (key, name))
                    elif key not in checked_member_keys:
                        failed_member.append("extra key %r in simple enum member %r" % (key, name))
                    else:
                        checked_value = checked_member_dict[key]
                        simple_value = simple_member_dict[key]
                        if checked_value != simple_value:
                            failed_member.append("%r:\n         %s\n         %s" % (
                                    key,
                                    "checked member -> %r" % (checked_value, ),
                                    "simple member  -> %r" % (simple_value, ),
                                    ))
            if failed_member:
                failed.append('%r member mismatch:\n      %s' % (
                        name, '\n      '.join(failed_member),
                        ))
        for method in (
                '__str__', '__repr__', '__reduce_ex__', '__format__',
                '__getnewargs_ex__', '__getnewargs__', '__reduce_ex__', '__reduce__'
            ):
            if method in simple_keys and method in checked_keys:
                # cannot compare functions, and it exists in both, so we're good
                continue
            elif method not in simple_keys and method not in checked_keys:
                # method is inherited -- check it out
                checked_method = getattr(checked_enum, method, None)
                simple_method = getattr(simple_enum, method, None)
                if hasattr(checked_method, '__func__'):
                    checked_method = checked_method.__func__
                    simple_method = simple_method.__func__
                if checked_method != simple_method:
                    failed.append("%r:  %-30s %s" % (
                            method,
                            "checked -> %r" % (checked_method, ),
                            "simple -> %r" % (simple_method, ),
                            ))
            else:
                # if the method existed in only one of the enums, it will have been caught
                # in the first checks above
                pass
    if failed:
        raise TypeError('enum mismatch:\n   %s' % '\n   '.join(failed))

def _old_convert_(etype, name, module, filter, source=None, *, boundary=None):
    """
    Create a new Enum subclass that replaces a collection of global constants
    """
    # convert all constants from source (or module) that pass filter() to
    # a new Enum called name, and export the enum and its members back to
    # module;
    # also, replace the __reduce_ex__ method so unpickling works in
    # previous Python versions
    module_globals = sys.modules[module].__dict__
    if source:
        source = source.__dict__
    else:
        source = module_globals
    # _value2member_map_ is populated in the same order every time
    # for a consistent reverse mapping of number to name when there
    # are multiple names for the same number.
    members = [
            (name, value)
            for name, value in source.items()
            if filter(name)]
    try:
        # sort by value
        members.sort(key=lambda t: (t[1], t[0]))
    except TypeError:
        # unless some values aren't comparable, in which case sort by name
        members.sort(key=lambda t: t[0])
    cls = etype(name, members, module=module, boundary=boundary or KEEP)
    return cls

_stdlib_enums = IntEnum, StrEnum, IntFlag



# Author: Steven J. Bethard <steven.bethard@gmail.com>.
# New maintainer as of 29 August 2019:  Raymond Hettinger <raymond.hettinger@gmail.com>

"""Command-line parsing library

This module is an optparse-inspired command-line parsing library that:

    - handles both optional and positional arguments
    - produces highly informative usage messages
    - supports parsers that dispatch to sub-parsers

The following is a simple usage example that sums integers from the
command-line and writes the result to a file::

    parser = argparse.ArgumentParser(
        description='sum the integers at the command line')
    parser.add_argument(
        'integers', metavar='int', nargs='+', type=int,
        help='an integer to be summed')
    parser.add_argument(
        '--log', default=sys.stdout, type=argparse.FileType('w'),
        help='the file where the sum should be written')
    args = parser.parse_args()
    args.log.write('%s' % sum(args.integers))
    args.log.close()

The module contains the following public classes:

    - ArgumentParser -- The main entry point for command-line parsing. As the
        example above shows, the add_argument() method is used to populate
        the parser with actions for optional and positional arguments. Then
        the parse_args() method is invoked to convert the args at the
        command-line into an object with attributes.

    - ArgumentError -- The exception raised by ArgumentParser objects when
        there are errors with the parser's actions. Errors raised while
        parsing the command-line are caught by ArgumentParser and emitted
        as command-line messages.

    - FileType -- A factory for defining types of files to be created. As the
        example above shows, instances of FileType are typically passed as
        the type= argument of add_argument() calls.

    - Action -- The base class for parser actions. Typically actions are
        selected by passing strings like 'store_true' or 'append_const' to
        the action= argument of add_argument(). However, for greater
        customization of ArgumentParser actions, subclasses of Action may
        be defined and passed as the action= argument.

    - HelpFormatter, RawDescriptionHelpFormatter, RawTextHelpFormatter,
        ArgumentDefaultsHelpFormatter -- Formatter classes which
        may be passed as the formatter_class= argument to the
        ArgumentParser constructor. HelpFormatter is the default,
        RawDescriptionHelpFormatter and RawTextHelpFormatter tell the parser
        not to change the formatting for help text, and
        ArgumentDefaultsHelpFormatter adds information about argument defaults
        to the help.

All other classes in this module are considered implementation details.
(Also note that HelpFormatter and RawDescriptionHelpFormatter are only
considered public as object names -- the API of the formatter objects is
still considered an implementation detail.)
"""

__version__ = '1.1'
__all__ = [
    'ArgumentParser',
    'ArgumentError',
    'ArgumentTypeError',
    'BooleanOptionalAction',
    'FileType',
    'HelpFormatter',
    'ArgumentDefaultsHelpFormatter',
    'RawDescriptionHelpFormatter',
    'RawTextHelpFormatter',
    'MetavarTypeHelpFormatter',
    'Namespace',
    'Action',
    'ONE_OR_MORE',
    'OPTIONAL',
    'PARSER',
    'REMAINDER',
    'SUPPRESS',
    'ZERO_OR_MORE',
]


import os as _os
import re as _re
import sys as _sys

from gettext import gettext as _, ngettext

SUPPRESS = '==SUPPRESS=='

OPTIONAL = '?'
ZERO_OR_MORE = '*'
ONE_OR_MORE = '+'
PARSER = 'A...'
REMAINDER = '...'
_UNRECOGNIZED_ARGS_ATTR = '_unrecognized_args'

# =============================
# Utility functions and classes
# =============================

class _AttributeHolder(object):
    """Abstract base class that provides __repr__.

    The __repr__ method returns a string in the format::
        ClassName(attr=name, attr=name, ...)
    The attributes are determined either by a class-level attribute,
    '_kwarg_names', or by inspecting the instance __dict__.
    """

    def __repr__(self):
        type_name = type(self).__name__
        arg_strings = []
        star_args = {}
        for arg in self._get_args():
            arg_strings.append(repr(arg))
        for name, value in self._get_kwargs():
            if name.isidentifier():
                arg_strings.append('%s=%r' % (name, value))
            else:
                star_args[name] = value
        if star_args:
            arg_strings.append('**%s' % repr(star_args))
        return '%s(%s)' % (type_name, ', '.join(arg_strings))

    def _get_kwargs(self):
        return list(self.__dict__.items())

    def _get_args(self):
        return []


def _copy_items(items):
    if items is None:
        return []
    # The copy module is used only in the 'append' and 'append_const'
    # actions, and it is needed only when the default value isn't a list.
    # Delay its import for speeding up the common case.
    if type(items) is list:
        return items[:]
    import copy
    return copy.copy(items)


# ===============
# Formatting Help
# ===============


class HelpFormatter(object):
    """Formatter for generating usage messages and argument help strings.

    Only the name of this class is considered a public API. All the methods
    provided by the class are considered an implementation detail.
    """

    def __init__(self,
                 prog,
                 indent_increment=2,
                 max_help_position=24,
                 width=None):

        # default setting for width
        if width is None:
            import shutil
            width = shutil.get_terminal_size().columns
            width -= 2

        self._prog = prog
        self._indent_increment = indent_increment
        self._max_help_position = min(max_help_position,
                                      max(width - 20, indent_increment * 2))
        self._width = width

        self._current_indent = 0
        self._level = 0
        self._action_max_length = 0

        self._root_section = self._Section(self, None)
        self._current_section = self._root_section

        self._whitespace_matcher = _re.compile(r'\s+', _re.ASCII)
        self._long_break_matcher = _re.compile(r'\n\n\n+')

    # ===============================
    # Section and indentation methods
    # ===============================

    def _indent(self):
        self._current_indent += self._indent_increment
        self._level += 1

    def _dedent(self):
        self._current_indent -= self._indent_increment
        assert self._current_indent >= 0, 'Indent decreased below 0.'
        self._level -= 1

    class _Section(object):

        def __init__(self, formatter, parent, heading=None):
            self.formatter = formatter
            self.parent = parent
            self.heading = heading
            self.items = []

        def format_help(self):
            # format the indented section
            if self.parent is not None:
                self.formatter._indent()
            join = self.formatter._join_parts
            item_help = join([func(*args) for func, args in self.items])
            if self.parent is not None:
                self.formatter._dedent()

            # return nothing if the section was empty
            if not item_help:
                return ''

            # add the heading if the section was non-empty
            if self.heading is not SUPPRESS and self.heading is not None:
                current_indent = self.formatter._current_indent
                heading_text = _('%(heading)s:') % dict(heading=self.heading)
                heading = '%*s%s\n' % (current_indent, '', heading_text)
            else:
                heading = ''

            # join the section-initial newline, the heading and the help
            return join(['\n', heading, item_help, '\n'])

    def _add_item(self, func, args):
        self._current_section.items.append((func, args))

    # ========================
    # Message building methods
    # ========================

    def start_section(self, heading):
        self._indent()
        section = self._Section(self, self._current_section, heading)
        self._add_item(section.format_help, [])
        self._current_section = section

    def end_section(self):
        self._current_section = self._current_section.parent
        self._dedent()

    def add_text(self, text):
        if text is not SUPPRESS and text is not None:
            self._add_item(self._format_text, [text])

    def add_usage(self, usage, actions, groups, prefix=None):
        if usage is not SUPPRESS:
            args = usage, actions, groups, prefix
            self._add_item(self._format_usage, args)

    def add_argument(self, action):
        if action.help is not SUPPRESS:

            # find all invocations
            get_invocation = self._format_action_invocation
            invocation_lengths = [len(get_invocation(action)) + self._current_indent]
            for subaction in self._iter_indented_subactions(action):
                invocation_lengths.append(len(get_invocation(subaction)) + self._current_indent)

            # update the maximum item length
            action_length = max(invocation_lengths)
            self._action_max_length = max(self._action_max_length,
                                          action_length)

            # add the item to the list
            self._add_item(self._format_action, [action])

    def add_arguments(self, actions):
        for action in actions:
            self.add_argument(action)

    # =======================
    # Help-formatting methods
    # =======================

    def format_help(self):
        help = self._root_section.format_help()
        if help:
            help = self._long_break_matcher.sub('\n\n', help)
            help = help.strip('\n') + '\n'
        return help

    def _join_parts(self, part_strings):
        return ''.join([part
                        for part in part_strings
                        if part and part is not SUPPRESS])

    def _format_usage(self, usage, actions, groups, prefix):
        if prefix is None:
            prefix = _('usage: ')

        # if usage is specified, use that
        if usage is not None:
            usage = usage % dict(prog=self._prog)

        # if no optionals or positionals are available, usage is just prog
        elif usage is None and not actions:
            usage = '%(prog)s' % dict(prog=self._prog)

        # if optionals and positionals are available, calculate usage
        elif usage is None:
            prog = '%(prog)s' % dict(prog=self._prog)

            # split optionals from positionals
            optionals = []
            positionals = []
            for action in actions:
                if action.option_strings:
                    optionals.append(action)
                else:
                    positionals.append(action)

            # build full usage string
            format = self._format_actions_usage
            action_usage = format(optionals + positionals, groups)
            usage = ' '.join([s for s in [prog, action_usage] if s])

            # wrap the usage parts if it's too long
            text_width = self._width - self._current_indent
            if len(prefix) + len(usage) > text_width:

                # break usage into wrappable parts
                # keep optionals and positionals together to preserve
                # mutually exclusive group formatting (gh-75949)
                all_actions = optionals + positionals
                parts, pos_start = self._get_actions_usage_parts_with_split(
                    all_actions, groups, len(optionals)
                )
                opt_parts = parts[:pos_start]
                pos_parts = parts[pos_start:]

                # helper for wrapping lines
                def get_lines(parts, indent, prefix=None):
                    lines = []
                    line = []
                    indent_length = len(indent)
                    if prefix is not None:
                        line_len = len(prefix) - 1
                    else:
                        line_len = indent_length - 1
                    for part in parts:
                        if line_len + 1 + len(part) > text_width and line:
                            lines.append(indent + ' '.join(line))
                            line = []
                            line_len = indent_length - 1
                        line.append(part)
                        line_len += len(part) + 1
                    if line:
                        lines.append(indent + ' '.join(line))
                    if prefix is not None:
                        lines[0] = lines[0][indent_length:]
                    return lines

                # if prog is short, follow it with optionals or positionals
                if len(prefix) + len(prog) <= 0.75 * text_width:
                    indent = ' ' * (len(prefix) + len(prog) + 1)
                    if opt_parts:
                        lines = get_lines([prog] + opt_parts, indent, prefix)
                        lines.extend(get_lines(pos_parts, indent))
                    elif pos_parts:
                        lines = get_lines([prog] + pos_parts, indent, prefix)
                    else:
                        lines = [prog]

                # if prog is long, put it on its own line
                else:
                    indent = ' ' * len(prefix)
                    parts = opt_parts + pos_parts
                    lines = get_lines(parts, indent)
                    if len(lines) > 1:
                        lines = []
                        lines.extend(get_lines(opt_parts, indent))
                        lines.extend(get_lines(pos_parts, indent))
                    lines = [prog] + lines

                # join lines into usage
                usage = '\n'.join(lines)

        # prefix with 'usage:'
        return '%s%s\n\n' % (prefix, usage)

    def _format_actions_usage(self, actions, groups):
        return ' '.join(self._get_actions_usage_parts(actions, groups))

    def _get_actions_usage_parts(self, actions, groups):
        parts, _ = self._get_actions_usage_parts_with_split(actions, groups)
        return parts

    def _get_actions_usage_parts_with_split(self, actions, groups, opt_count=None):
        """Get usage parts with split index for optionals/positionals.

        Returns (parts, pos_start) where pos_start is the index in parts
        where positionals begin. When opt_count is None, pos_start is None.
        This preserves mutually exclusive group formatting across the
        optionals/positionals boundary (gh-75949).
        """
        # find group indices and identify actions in groups
        group_actions = set()
        inserts = {}
        for group in groups:
            if not group._group_actions:
                raise ValueError(f'empty group {group}')

            if all(action.help is SUPPRESS for action in group._group_actions):
                continue

            try:
                start = actions.index(group._group_actions[0])
            except ValueError:
                continue
            else:
                end = start + len(group._group_actions)
                if actions[start:end] == group._group_actions:
                    group_actions.update(group._group_actions)
                    inserts[start, end] = group

        # collect all actions format strings
        parts = []
        for action in actions:

            # suppressed arguments are marked with None
            if action.help is SUPPRESS:
                part = None

            # produce all arg strings
            elif not action.option_strings:
                default = self._get_default_metavar_for_positional(action)
                part = self._format_args(action, default)

                # if it's in a group, strip the outer []
                if action in group_actions:
                    if part[0] == '[' and part[-1] == ']':
                        part = part[1:-1]

            # produce the first way to invoke the option in brackets
            else:
                option_string = action.option_strings[0]

                # if the Optional doesn't take a value, format is:
                #    -s or --long
                if action.nargs == 0:
                    part = action.format_usage()

                # if the Optional takes a value, format is:
                #    -s ARGS or --long ARGS
                else:
                    default = self._get_default_metavar_for_optional(action)
                    args_string = self._format_args(action, default)
                    part = '%s %s' % (option_string, args_string)

                # make it look optional if it's not required or in a group
                if not action.required and action not in group_actions:
                    part = '[%s]' % part

            # add the action string to the list
            parts.append(part)

        # group mutually exclusive actions
        inserted_separators_indices = set()
        for start, end in sorted(inserts, reverse=True):
            group = inserts[start, end]
            group_parts = [item for item in parts[start:end] if item is not None]
            group_size = len(group_parts)
            if group.required:
                open, close = "()" if group_size > 1 else ("", "")
            else:
                open, close = "[]"
            group_parts[0] = open + group_parts[0]
            group_parts[-1] = group_parts[-1] + close
            for i, part in enumerate(group_parts[:-1], start=start):
                # insert a separator if not already done in a nested group
                if i not in inserted_separators_indices:
                    parts[i] = part + ' |'
                    inserted_separators_indices.add(i)
            parts[start + group_size - 1] = group_parts[-1]
            for i in range(start + group_size, end):
                parts[i] = None

        # if opt_count is provided, calculate where positionals start in
        # the final parts list (for wrapping onto separate lines).
        # Count before filtering None entries since indices shift after.
        if opt_count is not None:
            pos_start = sum(1 for p in parts[:opt_count] if p is not None)
        else:
            pos_start = None

        # return the usage parts and split point (gh-75949)
        return [item for item in parts if item is not None], pos_start

    def _format_text(self, text):
        if '%(prog)' in text:
            text = text % dict(prog=self._prog)
        text_width = max(self._width - self._current_indent, 11)
        indent = ' ' * self._current_indent
        return self._fill_text(text, text_width, indent) + '\n\n'

    def _format_action(self, action):
        # determine the required width and the entry label
        help_position = min(self._action_max_length + 2,
                            self._max_help_position)
        help_width = max(self._width - help_position, 11)
        action_width = help_position - self._current_indent - 2
        action_header = self._format_action_invocation(action)

        # no help; start on same line and add a final newline
        if not action.help:
            tup = self._current_indent, '', action_header
            action_header = '%*s%s\n' % tup

        # short action name; start on the same line and pad two spaces
        elif len(action_header) <= action_width:
            tup = self._current_indent, '', action_width, action_header
            action_header = '%*s%-*s  ' % tup
            indent_first = 0

        # long action name; start on the next line
        else:
            tup = self._current_indent, '', action_header
            action_header = '%*s%s\n' % tup
            indent_first = help_position

        # collect the pieces of the action help
        parts = [action_header]

        # if there was help for the action, add lines of help text
        if action.help and action.help.strip():
            help_text = self._expand_help(action)
            if help_text:
                help_lines = self._split_lines(help_text, help_width)
                parts.append('%*s%s\n' % (indent_first, '', help_lines[0]))
                for line in help_lines[1:]:
                    parts.append('%*s%s\n' % (help_position, '', line))

        # or add a newline if the description doesn't end with one
        elif not action_header.endswith('\n'):
            parts.append('\n')

        # if there are any sub-actions, add their help as well
        for subaction in self._iter_indented_subactions(action):
            parts.append(self._format_action(subaction))

        # return a single string
        return self._join_parts(parts)

    def _format_action_invocation(self, action):
        if not action.option_strings:
            default = self._get_default_metavar_for_positional(action)
            return ' '.join(self._metavar_formatter(action, default)(1))

        else:

            # if the Optional doesn't take a value, format is:
            #    -s, --long
            if action.nargs == 0:
                return ', '.join(action.option_strings)

            # if the Optional takes a value, format is:
            #    -s, --long ARGS
            else:
                default = self._get_default_metavar_for_optional(action)
                args_string = self._format_args(action, default)
                return ', '.join(action.option_strings) + ' ' + args_string

    def _metavar_formatter(self, action, default_metavar):
        if action.metavar is not None:
            result = action.metavar
        elif action.choices is not None:
            result = '{%s}' % ','.join(map(str, action.choices))
        else:
            result = default_metavar

        def format(tuple_size):
            if isinstance(result, tuple):
                return result
            else:
                return (result, ) * tuple_size
        return format

    def _format_args(self, action, default_metavar):
        get_metavar = self._metavar_formatter(action, default_metavar)
        if action.nargs is None:
            result = '%s' % get_metavar(1)
        elif action.nargs == OPTIONAL:
            result = '[%s]' % get_metavar(1)
        elif action.nargs == ZERO_OR_MORE:
            metavar = get_metavar(1)
            if len(metavar) == 2:
                result = '[%s [%s ...]]' % metavar
            else:
                result = '[%s ...]' % metavar
        elif action.nargs == ONE_OR_MORE:
            result = '%s [%s ...]' % get_metavar(2)
        elif action.nargs == REMAINDER:
            result = '...'
        elif action.nargs == PARSER:
            result = '%s ...' % get_metavar(1)
        elif action.nargs == SUPPRESS:
            result = ''
        else:
            try:
                formats = ['%s' for _ in range(action.nargs)]
            except TypeError:
                raise ValueError("invalid nargs value") from None
            result = ' '.join(formats) % get_metavar(action.nargs)
        return result

    def _expand_help(self, action):
        params = dict(vars(action), prog=self._prog)
        for name in list(params):
            if params[name] is SUPPRESS:
                del params[name]
        for name in list(params):
            if hasattr(params[name], '__name__'):
                params[name] = params[name].__name__
        if params.get('choices') is not None:
            params['choices'] = ', '.join(map(str, params['choices']))
        return self._get_help_string(action) % params

    def _iter_indented_subactions(self, action):
        try:
            get_subactions = action._get_subactions
        except AttributeError:
            pass
        else:
            self._indent()
            yield from get_subactions()
            self._dedent()

    def _split_lines(self, text, width):
        text = self._whitespace_matcher.sub(' ', text).strip()
        # The textwrap module is used only for formatting help.
        # Delay its import for speeding up the common usage of argparse.
        import textwrap
        return textwrap.wrap(text, width)

    def _fill_text(self, text, width, indent):
        text = self._whitespace_matcher.sub(' ', text).strip()
        import textwrap
        return textwrap.fill(text, width,
                             initial_indent=indent,
                             subsequent_indent=indent)

    def _get_help_string(self, action):
        return action.help

    def _get_default_metavar_for_optional(self, action):
        return action.dest.upper()

    def _get_default_metavar_for_positional(self, action):
        return action.dest


class RawDescriptionHelpFormatter(HelpFormatter):
    """Help message formatter which retains any formatting in descriptions.

    Only the name of this class is considered a public API. All the methods
    provided by the class are considered an implementation detail.
    """

    def _fill_text(self, text, width, indent):
        return ''.join(indent + line for line in text.splitlines(keepends=True))


class RawTextHelpFormatter(RawDescriptionHelpFormatter):
    """Help message formatter which retains formatting of all help text.

    Only the name of this class is considered a public API. All the methods
    provided by the class are considered an implementation detail.
    """

    def _split_lines(self, text, width):
        return text.splitlines()


class ArgumentDefaultsHelpFormatter(HelpFormatter):
    """Help message formatter which adds default values to argument help.

    Only the name of this class is considered a public API. All the methods
    provided by the class are considered an implementation detail.
    """

    def _get_help_string(self, action):
        help = action.help
        if help is None:
            help = ''

        if (
            '%(default)' not in help
            and action.default is not SUPPRESS
            and not action.required
        ):
            defaulting_nargs = (OPTIONAL, ZERO_OR_MORE)
            if action.option_strings or action.nargs in defaulting_nargs:
                help += _(' (default: %(default)s)')
        return help



class MetavarTypeHelpFormatter(HelpFormatter):
    """Help message formatter which uses the argument 'type' as the default
    metavar value (instead of the argument 'dest')

    Only the name of this class is considered a public API. All the methods
    provided by the class are considered an implementation detail.
    """

    def _get_default_metavar_for_optional(self, action):
        return action.type.__name__

    def _get_default_metavar_for_positional(self, action):
        return action.type.__name__


# =====================
# Options and Arguments
# =====================

def _get_action_name(argument):
    if argument is None:
        return None
    elif argument.option_strings:
        return '/'.join(argument.option_strings)
    elif argument.metavar not in (None, SUPPRESS):
        metavar = argument.metavar
        if not isinstance(metavar, tuple):
            return metavar
        if argument.nargs == ZERO_OR_MORE and len(metavar) == 2:
            return '%s[, %s]' % metavar
        elif argument.nargs == ONE_OR_MORE:
            return '%s[, %s]' % metavar
        else:
            return ', '.join(metavar)
    elif argument.dest not in (None, SUPPRESS):
        return argument.dest
    elif argument.choices:
        return '{%s}' % ','.join(map(str, argument.choices))
    else:
        return None


class ArgumentError(Exception):
    """An error from creating or using an argument (optional or positional).

    The string value of this exception is the message, augmented with
    information about the argument that caused it.
    """

    def __init__(self, argument, message):
        self.argument_name = _get_action_name(argument)
        self.message = message

    def __str__(self):
        if self.argument_name is None:
            format = '%(message)s'
        else:
            format = _('argument %(argument_name)s: %(message)s')
        return format % dict(message=self.message,
                             argument_name=self.argument_name)


class ArgumentTypeError(Exception):
    """An error from trying to convert a command line string to a type."""
    pass


# ==============
# Action classes
# ==============

class Action(_AttributeHolder):
    """Information about how to convert command line strings to Python objects.

    Action objects are used by an ArgumentParser to represent the information
    needed to parse a single argument from one or more strings from the
    command line. The keyword arguments to the Action constructor are also
    all attributes of Action instances.

    Keyword Arguments:

        - option_strings -- A list of command-line option strings which
            should be associated with this action.

        - dest -- The name of the attribute to hold the created object(s)

        - nargs -- The number of command-line arguments that should be
            consumed. By default, one argument will be consumed and a single
            value will be produced.  Other values include:
                - N (an integer) consumes N arguments (and produces a list)
                - '?' consumes zero or one arguments
                - '*' consumes zero or more arguments (and produces a list)
                - '+' consumes one or more arguments (and produces a list)
            Note that the difference between the default and nargs=1 is that
            with the default, a single value will be produced, while with
            nargs=1, a list containing a single value will be produced.

        - const -- The value to be produced if the option is specified and the
            option uses an action that takes no values.

        - default -- The value to be produced if the option is not specified.

        - type -- A callable that accepts a single string argument, and
            returns the converted value.  The standard Python types str, int,
            float, and complex are useful examples of such callables.  If None,
            str is used.

        - choices -- A container of values that should be allowed. If not None,
            after a command-line argument has been converted to the appropriate
            type, an exception will be raised if it is not a member of this
            collection.

        - required -- True if the action must always be specified at the
            command line. This is only meaningful for optional command-line
            arguments.

        - help -- The help string describing the argument.

        - metavar -- The name to be used for the option's argument with the
            help string. If None, the 'dest' value will be used as the name.
    """

    def __init__(self,
                 option_strings,
                 dest,
                 nargs=None,
                 const=None,
                 default=None,
                 type=None,
                 choices=None,
                 required=False,
                 help=None,
                 metavar=None,
                 deprecated=False):
        self.option_strings = option_strings
        self.dest = dest
        self.nargs = nargs
        self.const = const
        self.default = default
        self.type = type
        self.choices = choices
        self.required = required
        self.help = help
        self.metavar = metavar
        self.deprecated = deprecated

    def _get_kwargs(self):
        names = [
            'option_strings',
            'dest',
            'nargs',
            'const',
            'default',
            'type',
            'choices',
            'required',
            'help',
            'metavar',
            'deprecated',
        ]
        return [(name, getattr(self, name)) for name in names]

    def format_usage(self):
        return self.option_strings[0]

    def __call__(self, parser, namespace, values, option_string=None):
        raise NotImplementedError(_('.__call__() not defined'))


# FIXME: remove together with `BooleanOptionalAction` deprecated arguments.
_deprecated_default = object()

class BooleanOptionalAction(Action):
    def __init__(self,
                 option_strings,
                 dest,
                 default=None,
                 type=_deprecated_default,
                 choices=_deprecated_default,
                 required=False,
                 help=None,
                 metavar=_deprecated_default,
                 deprecated=False):

        _option_strings = []
        for option_string in option_strings:
            _option_strings.append(option_string)

            if option_string.startswith('--'):
                option_string = '--no-' + option_string[2:]
                _option_strings.append(option_string)

        # We need `_deprecated` special value to ban explicit arguments that
        # match default value. Like:
        #   parser.add_argument('-f', action=BooleanOptionalAction, type=int)
        for field_name in ('type', 'choices', 'metavar'):
            if locals()[field_name] is not _deprecated_default:
                import warnings
                warnings._deprecated(
                    field_name,
                    "{name!r} is deprecated as of Python 3.12 and will be "
                    "removed in Python {remove}.",
                    remove=(3, 14))

        if type is _deprecated_default:
            type = None
        if choices is _deprecated_default:
            choices = None
        if metavar is _deprecated_default:
            metavar = None

        super().__init__(
            option_strings=_option_strings,
            dest=dest,
            nargs=0,
            default=default,
            type=type,
            choices=choices,
            required=required,
            help=help,
            metavar=metavar,
            deprecated=deprecated)


    def __call__(self, parser, namespace, values, option_string=None):
        if option_string in self.option_strings:
            setattr(namespace, self.dest, not option_string.startswith('--no-'))

    def format_usage(self):
        return ' | '.join(self.option_strings)


class _StoreAction(Action):

    def __init__(self,
                 option_strings,
                 dest,
                 nargs=None,
                 const=None,
                 default=None,
                 type=None,
                 choices=None,
                 required=False,
                 help=None,
                 metavar=None,
                 deprecated=False):
        if nargs == 0:
            raise ValueError('nargs for store actions must be != 0; if you '
                             'have nothing to store, actions such as store '
                             'true or store const may be more appropriate')
        if const is not None and nargs != OPTIONAL:
            raise ValueError('nargs must be %r to supply const' % OPTIONAL)
        super(_StoreAction, self).__init__(
            option_strings=option_strings,
            dest=dest,
            nargs=nargs,
            const=const,
            default=default,
            type=type,
            choices=choices,
            required=required,
            help=help,
            metavar=metavar,
            deprecated=deprecated)

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, values)


class _StoreConstAction(Action):

    def __init__(self,
                 option_strings,
                 dest,
                 const=None,
                 default=None,
                 required=False,
                 help=None,
                 metavar=None,
                 deprecated=False):
        super(_StoreConstAction, self).__init__(
            option_strings=option_strings,
            dest=dest,
            nargs=0,
            const=const,
            default=default,
            required=required,
            help=help,
            deprecated=deprecated)

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, self.const)


class _StoreTrueAction(_StoreConstAction):

    def __init__(self,
                 option_strings,
                 dest,
                 default=False,
                 required=False,
                 help=None,
                 deprecated=False):
        super(_StoreTrueAction, self).__init__(
            option_strings=option_strings,
            dest=dest,
            const=True,
            deprecated=deprecated,
            required=required,
            help=help,
            default=default)


class _StoreFalseAction(_StoreConstAction):

    def __init__(self,
                 option_strings,
                 dest,
                 default=True,
                 required=False,
                 help=None,
                 deprecated=False):
        super(_StoreFalseAction, self).__init__(
            option_strings=option_strings,
            dest=dest,
            const=False,
            default=default,
            required=required,
            help=help,
            deprecated=deprecated)


class _AppendAction(Action):

    def __init__(self,
                 option_strings,
                 dest,
                 nargs=None,
                 const=None,
                 default=None,
                 type=None,
                 choices=None,
                 required=False,
                 help=None,
                 metavar=None,
                 deprecated=False):
        if nargs == 0:
            raise ValueError('nargs for append actions must be != 0; if arg '
                             'strings are not supplying the value to append, '
                             'the append const action may be more appropriate')
        if const is not None and nargs != OPTIONAL:
            raise ValueError('nargs must be %r to supply const' % OPTIONAL)
        super(_AppendAction, self).__init__(
            option_strings=option_strings,
            dest=dest,
            nargs=nargs,
            const=const,
            default=default,
            type=type,
            choices=choices,
            required=required,
            help=help,
            metavar=metavar,
            deprecated=deprecated)

    def __call__(self, parser, namespace, values, option_string=None):
        items = getattr(namespace, self.dest, None)
        items = _copy_items(items)
        items.append(values)
        setattr(namespace, self.dest, items)


class _AppendConstAction(Action):

    def __init__(self,
                 option_strings,
                 dest,
                 const=None,
                 default=None,
                 required=False,
                 help=None,
                 metavar=None,
                 deprecated=False):
        super(_AppendConstAction, self).__init__(
            option_strings=option_strings,
            dest=dest,
            nargs=0,
            const=const,
            default=default,
            required=required,
            help=help,
            metavar=metavar,
            deprecated=deprecated)

    def __call__(self, parser, namespace, values, option_string=None):
        items = getattr(namespace, self.dest, None)
        items = _copy_items(items)
        items.append(self.const)
        setattr(namespace, self.dest, items)


class _CountAction(Action):

    def __init__(self,
                 option_strings,
                 dest,
                 default=None,
                 required=False,
                 help=None,
                 deprecated=False):
        super(_CountAction, self).__init__(
            option_strings=option_strings,
            dest=dest,
            nargs=0,
            default=default,
            required=required,
            help=help,
            deprecated=deprecated)

    def __call__(self, parser, namespace, values, option_string=None):
        count = getattr(namespace, self.dest, None)
        if count is None:
            count = 0
        setattr(namespace, self.dest, count + 1)


class _HelpAction(Action):

    def __init__(self,
                 option_strings,
                 dest=SUPPRESS,
                 default=SUPPRESS,
                 help=None,
                 deprecated=False):
        super(_HelpAction, self).__init__(
            option_strings=option_strings,
            dest=dest,
            default=default,
            nargs=0,
            help=help,
            deprecated=deprecated)

    def __call__(self, parser, namespace, values, option_string=None):
        parser.print_help()
        parser.exit()


class _VersionAction(Action):

    def __init__(self,
                 option_strings,
                 version=None,
                 dest=SUPPRESS,
                 default=SUPPRESS,
                 help=None,
                 deprecated=False):
        if help is None:
            help = _("show program's version number and exit")
        super(_VersionAction, self).__init__(
            option_strings=option_strings,
            dest=dest,
            default=default,
            nargs=0,
            help=help)
        self.version = version

    def __call__(self, parser, namespace, values, option_string=None):
        version = self.version
        if version is None:
            version = parser.version
        formatter = parser._get_formatter()
        formatter.add_text(version)
        parser._print_message(formatter.format_help(), _sys.stdout)
        parser.exit()


class _SubParsersAction(Action):

    class _ChoicesPseudoAction(Action):

        def __init__(self, name, aliases, help):
            metavar = dest = name
            if aliases:
                metavar += ' (%s)' % ', '.join(aliases)
            sup = super(_SubParsersAction._ChoicesPseudoAction, self)
            sup.__init__(option_strings=[], dest=dest, help=help,
                         metavar=metavar)

    def __init__(self,
                 option_strings,
                 prog,
                 parser_class,
                 dest=SUPPRESS,
                 required=False,
                 help=None,
                 metavar=None):

        self._prog_prefix = prog
        self._parser_class = parser_class
        self._name_parser_map = {}
        self._choices_actions = []
        self._deprecated = set()

        super(_SubParsersAction, self).__init__(
            option_strings=option_strings,
            dest=dest,
            nargs=PARSER,
            choices=self._name_parser_map,
            required=required,
            help=help,
            metavar=metavar)

    def add_parser(self, name, *, deprecated=False, **kwargs):
        # set prog from the existing prefix
        if kwargs.get('prog') is None:
            kwargs['prog'] = '%s %s' % (self._prog_prefix, name)

        aliases = kwargs.pop('aliases', ())

        if name in self._name_parser_map:
            raise ArgumentError(self, _('conflicting subparser: %s') % name)
        for alias in aliases:
            if alias in self._name_parser_map:
                raise ArgumentError(
                    self, _('conflicting subparser alias: %s') % alias)

        # create a pseudo-action to hold the choice help
        if 'help' in kwargs:
            help = kwargs.pop('help')
            choice_action = self._ChoicesPseudoAction(name, aliases, help)
            self._choices_actions.append(choice_action)

        # create the parser and add it to the map
        parser = self._parser_class(**kwargs)
        self._name_parser_map[name] = parser

        # make parser available under aliases also
        for alias in aliases:
            self._name_parser_map[alias] = parser

        if deprecated:
            self._deprecated.add(name)
            self._deprecated.update(aliases)

        return parser

    def _get_subactions(self):
        return self._choices_actions

    def __call__(self, parser, namespace, values, option_string=None):
        parser_name = values[0]
        arg_strings = values[1:]

        # set the parser name if requested
        if self.dest is not SUPPRESS:
            setattr(namespace, self.dest, parser_name)

        # select the parser
        try:
            subparser = self._name_parser_map[parser_name]
        except KeyError:
            args = {'parser_name': parser_name,
                    'choices': ', '.join(self._name_parser_map)}
            msg = _('unknown parser %(parser_name)r (choices: %(choices)s)') % args
            raise ArgumentError(self, msg)

        if parser_name in self._deprecated:
            parser._warning(_("command '%(parser_name)s' is deprecated") %
                            {'parser_name': parser_name})

        # parse all the remaining options into the namespace
        # store any unrecognized options on the object, so that the top
        # level parser can decide what to do with them

        # In case this subparser defines new defaults, we parse them
        # in a new namespace object and then update the original
        # namespace for the relevant parts.
        subnamespace, arg_strings = subparser.parse_known_args(arg_strings, None)
        for key, value in vars(subnamespace).items():
            setattr(namespace, key, value)

        if arg_strings:
            if not hasattr(namespace, _UNRECOGNIZED_ARGS_ATTR):
                setattr(namespace, _UNRECOGNIZED_ARGS_ATTR, [])
            getattr(namespace, _UNRECOGNIZED_ARGS_ATTR).extend(arg_strings)

class _ExtendAction(_AppendAction):
    def __call__(self, parser, namespace, values, option_string=None):
        items = getattr(namespace, self.dest, None)
        items = _copy_items(items)
        items.extend(values)
        setattr(namespace, self.dest, items)

# ==============
# Type classes
# ==============

class FileType(object):
    """Factory for creating file object types

    Instances of FileType are typically passed as type= arguments to the
    ArgumentParser add_argument() method.

    Keyword Arguments:
        - mode -- A string indicating how the file is to be opened. Accepts the
            same values as the builtin open() function.
        - bufsize -- The file's desired buffer size. Accepts the same values as
            the builtin open() function.
        - encoding -- The file's encoding. Accepts the same values as the
            builtin open() function.
        - errors -- A string indicating how encoding and decoding errors are to
            be handled. Accepts the same value as the builtin open() function.
    """

    def __init__(self, mode='r', bufsize=-1, encoding=None, errors=None):
        self._mode = mode
        self._bufsize = bufsize
        self._encoding = encoding
        self._errors = errors

    def __call__(self, string):
        # the special argument "-" means sys.std{in,out}
        if string == '-':
            if 'r' in self._mode:
                return _sys.stdin.buffer if 'b' in self._mode else _sys.stdin
            elif any(c in self._mode for c in 'wax'):
                return _sys.stdout.buffer if 'b' in self._mode else _sys.stdout
            else:
                msg = _('argument "-" with mode %r') % self._mode
                raise ValueError(msg)

        # all other arguments are used as file names
        try:
            return open(string, self._mode, self._bufsize, self._encoding,
                        self._errors)
        except OSError as e:
            args = {'filename': string, 'error': e}
            message = _("can't open '%(filename)s': %(error)s")
            raise ArgumentTypeError(message % args)

    def __repr__(self):
        args = self._mode, self._bufsize
        kwargs = [('encoding', self._encoding), ('errors', self._errors)]
        args_str = ', '.join([repr(arg) for arg in args if arg != -1] +
                             ['%s=%r' % (kw, arg) for kw, arg in kwargs
                              if arg is not None])
        return '%s(%s)' % (type(self).__name__, args_str)

# ===========================
# Optional and Positional Parsing
# ===========================

class Namespace(_AttributeHolder):
    """Simple object for storing attributes.

    Implements equality by attribute names and values, and provides a simple
    string representation.
    """

    def __init__(self, **kwargs):
        for name in kwargs:
            setattr(self, name, kwargs[name])

    def __eq__(self, other):
        if not isinstance(other, Namespace):
            return NotImplemented
        return vars(self) == vars(other)

    def __contains__(self, key):
        return key in self.__dict__


class _ActionsContainer(object):

    def __init__(self,
                 description,
                 prefix_chars,
                 argument_default,
                 conflict_handler):
        super(_ActionsContainer, self).__init__()

        self.description = description
        self.argument_default = argument_default
        self.prefix_chars = prefix_chars
        self.conflict_handler = conflict_handler

        # set up registries
        self._registries = {}

        # register actions
        self.register('action', None, _StoreAction)
        self.register('action', 'store', _StoreAction)
        self.register('action', 'store_const', _StoreConstAction)
        self.register('action', 'store_true', _StoreTrueAction)
        self.register('action', 'store_false', _StoreFalseAction)
        self.register('action', 'append', _AppendAction)
        self.register('action', 'append_const', _AppendConstAction)
        self.register('action', 'count', _CountAction)
        self.register('action', 'help', _HelpAction)
        self.register('action', 'version', _VersionAction)
        self.register('action', 'parsers', _SubParsersAction)
        self.register('action', 'extend', _ExtendAction)

        # raise an exception if the conflict handler is invalid
        self._get_handler()

        # action storage
        self._actions = []
        self._option_string_actions = {}

        # groups
        self._action_groups = []
        self._mutually_exclusive_groups = []

        # defaults storage
        self._defaults = {}

        # determines whether an "option" looks like a negative number
        self._negative_number_matcher = _re.compile(r'^-\d+$|^-\d*\.\d+$')

        # whether or not there are any optionals that look like negative
        # numbers -- uses a list so it can be shared and edited
        self._has_negative_number_optionals = []

    # ====================
    # Registration methods
    # ====================

    def register(self, registry_name, value, object):
        registry = self._registries.setdefault(registry_name, {})
        registry[value] = object

    def _registry_get(self, registry_name, value, default=None):
        return self._registries[registry_name].get(value, default)

    # ==================================
    # Namespace default accessor methods
    # ==================================

    def set_defaults(self, **kwargs):
        self._defaults.update(kwargs)

        # if these defaults match any existing arguments, replace
        # the previous default on the object with the new one
        for action in self._actions:
            if action.dest in kwargs:
                action.default = kwargs[action.dest]

    def get_default(self, dest):
        for action in self._actions:
            if action.dest == dest and action.default is not None:
                return action.default
        return self._defaults.get(dest, None)


    # =======================
    # Adding argument actions
    # =======================

    def add_argument(self, *args, **kwargs):
        """
        add_argument(dest, ..., name=value, ...)
        add_argument(option_string, option_string, ..., name=value, ...)
        """

        # if no positional args are supplied or only one is supplied and
        # it doesn't look like an option string, parse a positional
        # argument
        chars = self.prefix_chars
        if not args or len(args) == 1 and args[0][0] not in chars:
            if args and 'dest' in kwargs:
                raise ValueError('dest supplied twice for positional argument')
            kwargs = self._get_positional_kwargs(*args, **kwargs)

        # otherwise, we're adding an optional argument
        else:
            kwargs = self._get_optional_kwargs(*args, **kwargs)

        # if no default was supplied, use the parser-level default
        if 'default' not in kwargs:
            dest = kwargs['dest']
            if dest in self._defaults:
                kwargs['default'] = self._defaults[dest]
            elif self.argument_default is not None:
                kwargs['default'] = self.argument_default

        # create the action object, and add it to the parser
        action_class = self._pop_action_class(kwargs)
        if not callable(action_class):
            raise ValueError('unknown action "%s"' % (action_class,))
        action = action_class(**kwargs)

        # raise an error if the action type is not callable
        type_func = self._registry_get('type', action.type, action.type)
        if not callable(type_func):
            raise ValueError('%r is not callable' % (type_func,))

        if type_func is FileType:
            raise ValueError('%r is a FileType class object, instance of it'
                             ' must be passed' % (type_func,))

        # raise an error if the metavar does not match the type
        if hasattr(self, "_get_formatter"):
            try:
                self._get_formatter()._format_args(action, None)
            except TypeError:
                raise ValueError("length of metavar tuple does not match nargs")

        return self._add_action(action)

    def add_argument_group(self, *args, **kwargs):
        group = _ArgumentGroup(self, *args, **kwargs)
        self._action_groups.append(group)
        return group

    def add_mutually_exclusive_group(self, **kwargs):
        group = _MutuallyExclusiveGroup(self, **kwargs)
        self._mutually_exclusive_groups.append(group)
        return group

    def _add_action(self, action):
        # resolve any conflicts
        self._check_conflict(action)

        # add to actions list
        self._actions.append(action)
        action.container = self

        # index the action by any option strings it has
        for option_string in action.option_strings:
            self._option_string_actions[option_string] = action

        # set the flag if any option strings look like negative numbers
        for option_string in action.option_strings:
            if self._negative_number_matcher.match(option_string):
                if not self._has_negative_number_optionals:
                    self._has_negative_number_optionals.append(True)

        # return the created action
        return action

    def _remove_action(self, action):
        self._actions.remove(action)

    def _add_container_actions(self, container):
        # collect groups by titles
        title_group_map = {}
        for group in self._action_groups:
            if group.title in title_group_map:
                # This branch could happen if a derived class added
                # groups with duplicated titles in __init__
                msg = _('cannot merge actions - two groups are named %r')
                raise ValueError(msg % (group.title))
            title_group_map[group.title] = group

        # map each action to its group
        group_map = {}
        for group in container._action_groups:

            # if a group with the title exists, use that, otherwise
            # create a new group matching the container's group
            if group.title not in title_group_map:
                title_group_map[group.title] = self.add_argument_group(
                    title=group.title,
                    description=group.description,
                    conflict_handler=group.conflict_handler)

            # map the actions to their new group
            for action in group._group_actions:
                group_map[action] = title_group_map[group.title]

        # add container's mutually exclusive groups
        # NOTE: if add_mutually_exclusive_group ever gains title= and
        # description= then this code will need to be expanded as above
        for group in container._mutually_exclusive_groups:
            if group._container is container:
                cont = self
            else:
                cont = title_group_map[group._container.title]
            mutex_group = cont.add_mutually_exclusive_group(
                required=group.required)

            # map the actions to their new mutex group
            for action in group._group_actions:
                group_map[action] = mutex_group

        # add all actions to this container or their group
        for action in container._actions:
            group_map.get(action, self)._add_action(action)

    def _get_positional_kwargs(self, dest, **kwargs):
        # make sure required is not specified
        if 'required' in kwargs:
            msg = _("'required' is an invalid argument for positionals")
            raise TypeError(msg)

        # mark positional arguments as required if at least one is
        # always required
        nargs = kwargs.get('nargs')
        if nargs not in [OPTIONAL, ZERO_OR_MORE, REMAINDER, SUPPRESS, 0]:
            kwargs['required'] = True

        # return the keyword arguments with no option strings
        return dict(kwargs, dest=dest, option_strings=[])

    def _get_optional_kwargs(self, *args, **kwargs):
        # determine short and long option strings
        option_strings = []
        long_option_strings = []
        for option_string in args:
            # error on strings that don't start with an appropriate prefix
            if not option_string[0] in self.prefix_chars:
                args = {'option': option_string,
                        'prefix_chars': self.prefix_chars}
                msg = _('invalid option string %(option)r: '
                        'must start with a character %(prefix_chars)r')
                raise ValueError(msg % args)

            # strings starting with two prefix characters are long options
            option_strings.append(option_string)
            if len(option_string) > 1 and option_string[1] in self.prefix_chars:
                long_option_strings.append(option_string)

        # infer destination, '--foo-bar' -> 'foo_bar' and '-x' -> 'x'
        dest = kwargs.pop('dest', None)
        if dest is None:
            if long_option_strings:
                dest_option_string = long_option_strings[0]
            else:
                dest_option_string = option_strings[0]
            dest = dest_option_string.lstrip(self.prefix_chars)
            if not dest:
                msg = _('dest= is required for options like %r')
                raise ValueError(msg % option_string)
            dest = dest.replace('-', '_')

        # return the updated keyword arguments
        return dict(kwargs, dest=dest, option_strings=option_strings)

    def _pop_action_class(self, kwargs, default=None):
        action = kwargs.pop('action', default)
        return self._registry_get('action', action, action)

    def _get_handler(self):
        # determine function from conflict handler string
        handler_func_name = '_handle_conflict_%s' % self.conflict_handler
        try:
            return getattr(self, handler_func_name)
        except AttributeError:
            msg = _('invalid conflict_resolution value: %r')
            raise ValueError(msg % self.conflict_handler)

    def _check_conflict(self, action):

        # find all options that conflict with this option
        confl_optionals = []
        for option_string in action.option_strings:
            if option_string in self._option_string_actions:
                confl_optional = self._option_string_actions[option_string]
                confl_optionals.append((option_string, confl_optional))

        # resolve any conflicts
        if confl_optionals:
            conflict_handler = self._get_handler()
            conflict_handler(action, confl_optionals)

    def _handle_conflict_error(self, action, conflicting_actions):
        message = ngettext('conflicting option string: %s',
                           'conflicting option strings: %s',
                           len(conflicting_actions))
        conflict_string = ', '.join([option_string
                                     for option_string, action
                                     in conflicting_actions])
        raise ArgumentError(action, message % conflict_string)

    def _handle_conflict_resolve(self, action, conflicting_actions):

        # remove all conflicting options
        for option_string, action in conflicting_actions:

            # remove the conflicting option
            action.option_strings.remove(option_string)
            self._option_string_actions.pop(option_string, None)

            # if the option now has no option string, remove it from the
            # container holding it
            if not action.option_strings:
                action.container._remove_action(action)


class _ArgumentGroup(_ActionsContainer):

    def __init__(self, container, title=None, description=None, **kwargs):
        # add any missing keyword arguments by checking the container
        update = kwargs.setdefault
        update('conflict_handler', container.conflict_handler)
        update('prefix_chars', container.prefix_chars)
        update('argument_default', container.argument_default)
        super_init = super(_ArgumentGroup, self).__init__
        super_init(description=description, **kwargs)

        # group attributes
        self.title = title
        self._group_actions = []

        # share most attributes with the container
        self._registries = container._registries
        self._actions = container._actions
        self._option_string_actions = container._option_string_actions
        self._defaults = container._defaults
        self._has_negative_number_optionals = \
            container._has_negative_number_optionals
        self._mutually_exclusive_groups = container._mutually_exclusive_groups

    def _add_action(self, action):
        action = super(_ArgumentGroup, self)._add_action(action)
        self._group_actions.append(action)
        return action

    def _remove_action(self, action):
        super(_ArgumentGroup, self)._remove_action(action)
        self._group_actions.remove(action)

    def add_argument_group(self, *args, **kwargs):
        import warnings
        warnings.warn(
            "Nesting argument groups is deprecated.",
            category=DeprecationWarning,
            stacklevel=2
        )
        return super().add_argument_group(*args, **kwargs)


class _MutuallyExclusiveGroup(_ArgumentGroup):

    def __init__(self, container, required=False):
        super(_MutuallyExclusiveGroup, self).__init__(container)
        self.required = required
        self._container = container

    def _add_action(self, action):
        if action.required:
            msg = _('mutually exclusive arguments must be optional')
            raise ValueError(msg)
        action = self._container._add_action(action)
        self._group_actions.append(action)
        return action

    def _remove_action(self, action):
        self._container._remove_action(action)
        self._group_actions.remove(action)

    def add_mutually_exclusive_group(self, *args, **kwargs):
        import warnings
        warnings.warn(
            "Nesting mutually exclusive groups is deprecated.",
            category=DeprecationWarning,
            stacklevel=2
        )
        return super().add_mutually_exclusive_group(*args, **kwargs)


class ArgumentParser(_AttributeHolder, _ActionsContainer):
    """Object for parsing command line strings into Python objects.

    Keyword Arguments:
        - prog -- The name of the program (default:
            ``os.path.basename(sys.argv[0])``)
        - usage -- A usage message (default: auto-generated from arguments)
        - description -- A description of what the program does
        - epilog -- Text following the argument descriptions
        - parents -- Parsers whose arguments should be copied into this one
        - formatter_class -- HelpFormatter class for printing help messages
        - prefix_chars -- Characters that prefix optional arguments
        - fromfile_prefix_chars -- Characters that prefix files containing
            additional arguments
        - argument_default -- The default value for all arguments
        - conflict_handler -- String indicating how to handle conflicts
        - add_help -- Add a -h/-help option
        - allow_abbrev -- Allow long options to be abbreviated unambiguously
        - exit_on_error -- Determines whether or not ArgumentParser exits with
            error info when an error occurs
    """

    def __init__(self,
                 prog=None,
                 usage=None,
                 description=None,
                 epilog=None,
                 parents=[],
                 formatter_class=HelpFormatter,
                 prefix_chars='-',
                 fromfile_prefix_chars=None,
                 argument_default=None,
                 conflict_handler='error',
                 add_help=True,
                 allow_abbrev=True,
                 exit_on_error=True):

        superinit = super(ArgumentParser, self).__init__
        superinit(description=description,
                  prefix_chars=prefix_chars,
                  argument_default=argument_default,
                  conflict_handler=conflict_handler)

        # default setting for prog
        if prog is None:
            prog = _os.path.basename(_sys.argv[0])

        self.prog = prog
        self.usage = usage
        self.epilog = epilog
        self.formatter_class = formatter_class
        self.fromfile_prefix_chars = fromfile_prefix_chars
        self.add_help = add_help
        self.allow_abbrev = allow_abbrev
        self.exit_on_error = exit_on_error

        add_group = self.add_argument_group
        self._positionals = add_group(_('positional arguments'))
        self._optionals = add_group(_('options'))
        self._subparsers = None

        # register types
        def identity(string):
            return string
        self.register('type', None, identity)

        # add help argument if necessary
        # (using explicit default to override global argument_default)
        default_prefix = '-' if '-' in prefix_chars else prefix_chars[0]
        if self.add_help:
            self.add_argument(
                default_prefix+'h', default_prefix*2+'help',
                action='help', default=SUPPRESS,
                help=_('show this help message and exit'))

        # add parent arguments and defaults
        for parent in parents:
            if not isinstance(parent, ArgumentParser):
                raise TypeError('parents must be a list of ArgumentParser')
            self._add_container_actions(parent)
            defaults = parent._defaults
            self._defaults.update(defaults)

    # =======================
    # Pretty __repr__ methods
    # =======================

    def _get_kwargs(self):
        names = [
            'prog',
            'usage',
            'description',
            'formatter_class',
            'conflict_handler',
            'add_help',
        ]
        return [(name, getattr(self, name)) for name in names]

    # ==================================
    # Optional/Positional adding methods
    # ==================================

    def add_subparsers(self, **kwargs):
        if self._subparsers is not None:
            raise ArgumentError(None, _('cannot have multiple subparser arguments'))

        # add the parser class to the arguments if it's not present
        kwargs.setdefault('parser_class', type(self))

        if 'title' in kwargs or 'description' in kwargs:
            title = kwargs.pop('title', _('subcommands'))
            description = kwargs.pop('description', None)
            self._subparsers = self.add_argument_group(title, description)
        else:
            self._subparsers = self._positionals

        # prog defaults to the usage message of this parser, skipping
        # optional arguments and with no "usage:" prefix
        if kwargs.get('prog') is None:
            formatter = self._get_formatter()
            positionals = self._get_positional_actions()
            groups = self._mutually_exclusive_groups
            formatter.add_usage(self.usage, positionals, groups, '')
            kwargs['prog'] = formatter.format_help().strip()

        # create the parsers action and add it to the positionals list
        parsers_class = self._pop_action_class(kwargs, 'parsers')
        action = parsers_class(option_strings=[], **kwargs)
        self._subparsers._add_action(action)

        # return the created parsers action
        return action

    def _add_action(self, action):
        if action.option_strings:
            self._optionals._add_action(action)
        else:
            self._positionals._add_action(action)
        return action

    def _get_optional_actions(self):
        return [action
                for action in self._actions
                if action.option_strings]

    def _get_positional_actions(self):
        return [action
                for action in self._actions
                if not action.option_strings]

    # =====================================
    # Command line argument parsing methods
    # =====================================

    def parse_args(self, args=None, namespace=None):
        args, argv = self.parse_known_args(args, namespace)
        if argv:
            msg = _('unrecognized arguments: %s') % ' '.join(argv)
            if self.exit_on_error:
                self.error(msg)
            else:
                raise ArgumentError(None, msg)
        return args

    def parse_known_args(self, args=None, namespace=None):
        return self._parse_known_args2(args, namespace, intermixed=False)

    def _parse_known_args2(self, args, namespace, intermixed):
        if args is None:
            # args default to the system args
            args = _sys.argv[1:]
        else:
            # make sure that args are mutable
            args = list(args)

        # default Namespace built from parser defaults
        if namespace is None:
            namespace = Namespace()

        # add any action defaults that aren't present
        for action in self._actions:
            if action.dest is not SUPPRESS:
                if not hasattr(namespace, action.dest):
                    if action.default is not SUPPRESS:
                        setattr(namespace, action.dest, action.default)

        # add any parser defaults that aren't present
        for dest in self._defaults:
            if not hasattr(namespace, dest):
                setattr(namespace, dest, self._defaults[dest])

        # parse the arguments and exit if there are any errors
        if self.exit_on_error:
            try:
                namespace, args = self._parse_known_args(args, namespace, intermixed)
            except ArgumentError as err:
                self.error(str(err))
        else:
            namespace, args = self._parse_known_args(args, namespace, intermixed)

        if hasattr(namespace, _UNRECOGNIZED_ARGS_ATTR):
            args.extend(getattr(namespace, _UNRECOGNIZED_ARGS_ATTR))
            delattr(namespace, _UNRECOGNIZED_ARGS_ATTR)
        return namespace, args

    def _parse_known_args(self, arg_strings, namespace, intermixed):
        # replace arg strings that are file references
        if self.fromfile_prefix_chars is not None:
            arg_strings = self._read_args_from_files(arg_strings)

        # map all mutually exclusive arguments to the other arguments
        # they can't occur with
        action_conflicts = {}
        for mutex_group in self._mutually_exclusive_groups:
            group_actions = mutex_group._group_actions
            for i, mutex_action in enumerate(mutex_group._group_actions):
                conflicts = action_conflicts.setdefault(mutex_action, [])
                conflicts.extend(group_actions[:i])
                conflicts.extend(group_actions[i + 1:])

        # find all option indices, and determine the arg_string_pattern
        # which has an 'O' if there is an option at an index,
        # an 'A' if there is an argument, or a '-' if there is a '--'
        option_string_indices = {}
        arg_string_pattern_parts = []
        arg_strings_iter = iter(arg_strings)
        for i, arg_string in enumerate(arg_strings_iter):

            # all args after -- are non-options
            if arg_string == '--':
                arg_string_pattern_parts.append('-')
                for arg_string in arg_strings_iter:
                    arg_string_pattern_parts.append('A')

            # otherwise, add the arg to the arg strings
            # and note the index if it was an option
            else:
                option_tuples = self._parse_optional(arg_string)
                if option_tuples is None:
                    pattern = 'A'
                else:
                    option_string_indices[i] = option_tuples
                    pattern = 'O'
                arg_string_pattern_parts.append(pattern)

        # join the pieces together to form the pattern
        arg_strings_pattern = ''.join(arg_string_pattern_parts)

        # converts arg strings to the appropriate and then takes the action
        seen_actions = set()
        seen_non_default_actions = set()
        warned = set()

        def take_action(action, argument_strings, option_string=None):
            seen_actions.add(action)
            argument_values = self._get_values(action, argument_strings)

            # error if this argument is not allowed with other previously
            # seen arguments
            if action.option_strings or argument_strings:
                seen_non_default_actions.add(action)
                for conflict_action in action_conflicts.get(action, []):
                    if conflict_action in seen_non_default_actions:
                        msg = _('not allowed with argument %s')
                        action_name = _get_action_name(conflict_action)
                        raise ArgumentError(action, msg % action_name)

            # take the action if we didn't receive a SUPPRESS value
            # (e.g. from a default)
            if argument_values is not SUPPRESS:
                action(self, namespace, argument_values, option_string)

        # function to convert arg_strings into an optional action
        def consume_optional(start_index):

            # get the optional identified at this index
            option_tuples = option_string_indices[start_index]
            # if multiple actions match, the option string was ambiguous
            if len(option_tuples) > 1:
                options = ', '.join([option_string
                    for action, option_string, sep, explicit_arg in option_tuples])
                args = {'option': arg_strings[start_index], 'matches': options}
                msg = _('ambiguous option: %(option)s could match %(matches)s')
                raise ArgumentError(None, msg % args)

            action, option_string, sep, explicit_arg = option_tuples[0]

            # identify additional optionals in the same arg string
            # (e.g. -xyz is the same as -x -y -z if no args are required)
            match_argument = self._match_argument
            action_tuples = []
            while True:

                # if we found no optional action, skip it
                if action is None:
                    extras.append(arg_strings[start_index])
                    extras_pattern.append('O')
                    return start_index + 1

                # if there is an explicit argument, try to match the
                # optional's string arguments to only this
                if explicit_arg is not None:
                    arg_count = match_argument(action, 'A')

                    # if the action is a single-dash option and takes no
                    # arguments, try to parse more single-dash options out
                    # of the tail of the option string
                    chars = self.prefix_chars
                    if (
                        arg_count == 0
                        and option_string[1] not in chars
                        and explicit_arg != ''
                    ):
                        if sep or explicit_arg[0] in chars:
                            msg = _('ignored explicit argument %r')
                            raise ArgumentError(action, msg % explicit_arg)
                        action_tuples.append((action, [], option_string))
                        char = option_string[0]
                        option_string = char + explicit_arg[0]
                        optionals_map = self._option_string_actions
                        if option_string in optionals_map:
                            action = optionals_map[option_string]
                            explicit_arg = explicit_arg[1:]
                            if not explicit_arg:
                                sep = explicit_arg = None
                            elif explicit_arg[0] == '=':
                                sep = '='
                                explicit_arg = explicit_arg[1:]
                            else:
                                sep = ''
                        else:
                            extras.append(char + explicit_arg)
                            extras_pattern.append('O')
                            stop = start_index + 1
                            break
                    # if the action expect exactly one argument, we've
                    # successfully matched the option; exit the loop
                    elif arg_count == 1:
                        stop = start_index + 1
                        args = [explicit_arg]
                        action_tuples.append((action, args, option_string))
                        break

                    # error if a double-dash option did not use the
                    # explicit argument
                    else:
                        msg = _('ignored explicit argument %r')
                        raise ArgumentError(action, msg % explicit_arg)

                # if there is no explicit argument, try to match the
                # optional's string arguments with the following strings
                # if successful, exit the loop
                else:
                    start = start_index + 1
                    selected_patterns = arg_strings_pattern[start:]
                    arg_count = match_argument(action, selected_patterns)
                    stop = start + arg_count
                    args = arg_strings[start:stop]
                    action_tuples.append((action, args, option_string))
                    break

            # add the Optional to the list and return the index at which
            # the Optional's string args stopped
            assert action_tuples
            for action, args, option_string in action_tuples:
                if action.deprecated and option_string not in warned:
                    self._warning(_("option '%(option)s' is deprecated") %
                                  {'option': option_string})
                    warned.add(option_string)
                take_action(action, args, option_string)
            return stop

        # the list of Positionals left to be parsed; this is modified
        # by consume_positionals()
        positionals = self._get_positional_actions()

        # function to convert arg_strings into positional actions
        def consume_positionals(start_index):
            # match as many Positionals as possible
            match_partial = self._match_arguments_partial
            selected_pattern = arg_strings_pattern[start_index:]
            arg_counts = match_partial(positionals, selected_pattern)

            # slice off the appropriate arg strings for each Positional
            # and add the Positional and its args to the list
            for action, arg_count in zip(positionals, arg_counts):
                args = arg_strings[start_index: start_index + arg_count]
                # Strip out the first '--' if it is not in REMAINDER arg.
                if action.nargs == PARSER:
                    if arg_strings_pattern[start_index] == '-':
                        assert args[0] == '--'
                        args.remove('--')
                elif action.nargs != REMAINDER:
                    if (arg_strings_pattern.find('-', start_index,
                                                 start_index + arg_count) >= 0):
                        args.remove('--')
                start_index += arg_count
                if args and action.deprecated and action.dest not in warned:
                    self._warning(_("argument '%(argument_name)s' is deprecated") %
                                  {'argument_name': action.dest})
                    warned.add(action.dest)
                take_action(action, args)

            # slice off the Positionals that we just parsed and return the
            # index at which the Positionals' string args stopped
            positionals[:] = positionals[len(arg_counts):]
            return start_index

        # consume Positionals and Optionals alternately, until we have
        # passed the last option string
        extras = []
        extras_pattern = []
        start_index = 0
        if option_string_indices:
            max_option_string_index = max(option_string_indices)
        else:
            max_option_string_index = -1
        while start_index <= max_option_string_index:

            # consume any Positionals preceding the next option
            next_option_string_index = start_index
            while next_option_string_index <= max_option_string_index:
                if next_option_string_index in option_string_indices:
                    break
                next_option_string_index += 1
            if not intermixed and start_index != next_option_string_index:
                positionals_end_index = consume_positionals(start_index)

                # only try to parse the next optional if we didn't consume
                # the option string during the positionals parsing
                if positionals_end_index > start_index:
                    start_index = positionals_end_index
                    continue
                else:
                    start_index = positionals_end_index

            # if we consumed all the positionals we could and we're not
            # at the index of an option string, there were extra arguments
            if start_index not in option_string_indices:
                strings = arg_strings[start_index:next_option_string_index]
                extras.extend(strings)
                extras_pattern.extend(arg_strings_pattern[start_index:next_option_string_index])
                start_index = next_option_string_index

            # consume the next optional and any arguments for it
            start_index = consume_optional(start_index)

        if not intermixed:
            # consume any positionals following the last Optional
            stop_index = consume_positionals(start_index)

            # if we didn't consume all the argument strings, there were extras
            extras.extend(arg_strings[stop_index:])
        else:
            extras.extend(arg_strings[start_index:])
            extras_pattern.extend(arg_strings_pattern[start_index:])
            extras_pattern = ''.join(extras_pattern)
            assert len(extras_pattern) == len(extras)
            # consume all positionals
            arg_strings = [s for s, c in zip(extras, extras_pattern) if c != 'O']
            arg_strings_pattern = extras_pattern.replace('O', '')
            stop_index = consume_positionals(0)
            # leave unknown optionals and non-consumed positionals in extras
            for i, c in enumerate(extras_pattern):
                if not stop_index:
                    break
                if c != 'O':
                    stop_index -= 1
                    extras[i] = None
            extras = [s for s in extras if s is not None]

        # make sure all required actions were present and also convert
        # action defaults which were not given as arguments
        required_actions = []
        for action in self._actions:
            if action not in seen_actions:
                if action.required:
                    required_actions.append(_get_action_name(action))
                else:
                    # Convert action default now instead of doing it before
                    # parsing arguments to avoid calling convert functions
                    # twice (which may fail) if the argument was given, but
                    # only if it was defined already in the namespace
                    if (action.default is not None and
                        isinstance(action.default, str) and
                        hasattr(namespace, action.dest) and
                        action.default is getattr(namespace, action.dest)):
                        setattr(namespace, action.dest,
                                self._get_value(action, action.default))

        if required_actions:
            raise ArgumentError(None, _('the following arguments are required: %s') %
                       ', '.join(required_actions))

        # make sure all required groups had one option present
        for group in self._mutually_exclusive_groups:
            if group.required:
                for action in group._group_actions:
                    if action in seen_non_default_actions:
                        break

                # if no actions were used, report the error
                else:
                    names = [_get_action_name(action)
                             for action in group._group_actions
                             if action.help is not SUPPRESS]
                    msg = _('one of the arguments %s is required')
                    raise ArgumentError(None, msg % ' '.join(names))

        # return the updated namespace and the extra arguments
        return namespace, extras

    def _read_args_from_files(self, arg_strings):
        # expand arguments referencing files
        new_arg_strings = []
        for arg_string in arg_strings:

            # for regular arguments, just add them back into the list
            if not arg_string or arg_string[0] not in self.fromfile_prefix_chars:
                new_arg_strings.append(arg_string)

            # replace arguments referencing files with the file content
            else:
                try:
                    with open(arg_string[1:],
                              encoding=_sys.getfilesystemencoding(),
                              errors=_sys.getfilesystemencodeerrors()) as args_file:
                        arg_strings = []
                        for arg_line in args_file.read().splitlines():
                            for arg in self.convert_arg_line_to_args(arg_line):
                                arg_strings.append(arg)
                        arg_strings = self._read_args_from_files(arg_strings)
                        new_arg_strings.extend(arg_strings)
                except OSError as err:
                    raise ArgumentError(None, str(err))

        # return the modified argument list
        return new_arg_strings

    def convert_arg_line_to_args(self, arg_line):
        return [arg_line]

    def _match_argument(self, action, arg_strings_pattern):
        # match the pattern for this action to the arg strings
        nargs_pattern = self._get_nargs_pattern(action)
        match = _re.match(nargs_pattern, arg_strings_pattern)

        # raise an exception if we weren't able to find a match
        if match is None:
            nargs_errors = {
                None: _('expected one argument'),
                OPTIONAL: _('expected at most one argument'),
                ONE_OR_MORE: _('expected at least one argument'),
            }
            msg = nargs_errors.get(action.nargs)
            if msg is None:
                msg = ngettext('expected %s argument',
                               'expected %s arguments',
                               action.nargs) % action.nargs
            raise ArgumentError(action, msg)

        # return the number of arguments matched
        return len(match.group(1))

    def _match_arguments_partial(self, actions, arg_strings_pattern):
        # progressively shorten the actions list by slicing off the
        # final actions until we find a match
        for i in range(len(actions), 0, -1):
            actions_slice = actions[:i]
            pattern = ''.join([self._get_nargs_pattern(action)
                               for action in actions_slice])
            match = _re.match(pattern, arg_strings_pattern)
            if match is not None:
                result = [len(string) for string in match.groups()]
                if (match.end() < len(arg_strings_pattern)
                    and arg_strings_pattern[match.end()] == 'O'):
                    while result and not result[-1]:
                        del result[-1]
                return result
        return []

    def _parse_optional(self, arg_string):
        # if it's an empty string, it was meant to be a positional
        if not arg_string:
            return None

        # if it doesn't start with a prefix, it was meant to be positional
        if not arg_string[0] in self.prefix_chars:
            return None

        # if the option string is present in the parser, return the action
        if arg_string in self._option_string_actions:
            action = self._option_string_actions[arg_string]
            return [(action, arg_string, None, None)]

        # if it's just a single character, it was meant to be positional
        if len(arg_string) == 1:
            return None

        # if the option string before the "=" is present, return the action
        option_string, sep, explicit_arg = arg_string.partition('=')
        if sep and option_string in self._option_string_actions:
            action = self._option_string_actions[option_string]
            return [(action, option_string, sep, explicit_arg)]

        # search through all possible prefixes of the option string
        # and all actions in the parser for possible interpretations
        option_tuples = self._get_option_tuples(arg_string)

        if option_tuples:
            return option_tuples

        # if it was not found as an option, but it looks like a negative
        # number, it was meant to be positional
        # unless there are negative-number-like options
        if self._negative_number_matcher.match(arg_string):
            if not self._has_negative_number_optionals:
                return None

        # if it contains a space, it was meant to be a positional
        if ' ' in arg_string:
            return None

        # it was meant to be an optional but there is no such option
        # in this parser (though it might be a valid option in a subparser)
        return [(None, arg_string, None, None)]

    def _get_option_tuples(self, option_string):
        result = []

        # option strings starting with two prefix characters are only
        # split at the '='
        chars = self.prefix_chars
        if option_string[0] in chars and option_string[1] in chars:
            if self.allow_abbrev:
                option_prefix, sep, explicit_arg = option_string.partition('=')
                if not sep:
                    sep = explicit_arg = None
                for option_string in self._option_string_actions:
                    if option_string.startswith(option_prefix):
                        action = self._option_string_actions[option_string]
                        tup = action, option_string, sep, explicit_arg
                        result.append(tup)

        # single character options can be concatenated with their arguments
        # but multiple character options always have to have their argument
        # separate
        elif option_string[0] in chars and option_string[1] not in chars:
            option_prefix, sep, explicit_arg = option_string.partition('=')
            if not sep:
                sep = explicit_arg = None
            short_option_prefix = option_string[:2]
            short_explicit_arg = option_string[2:]

            for option_string in self._option_string_actions:
                if option_string == short_option_prefix:
                    action = self._option_string_actions[option_string]
                    tup = action, option_string, '', short_explicit_arg
                    result.append(tup)
                elif self.allow_abbrev and option_string.startswith(option_prefix):
                    action = self._option_string_actions[option_string]
                    tup = action, option_string, sep, explicit_arg
                    result.append(tup)

        # shouldn't ever get here
        else:
            raise ArgumentError(None, _('unexpected option string: %s') % option_string)

        # return the collected option tuples
        return result

    def _get_nargs_pattern(self, action):
        # in all examples below, we have to allow for '--' args
        # which are represented as '-' in the pattern
        nargs = action.nargs
        # if this is an optional action, -- is not allowed
        option = action.option_strings

        # the default (None) is assumed to be a single argument
        if nargs is None:
            nargs_pattern = '([A])' if option else '(-*A-*)'

        # allow zero or one arguments
        elif nargs == OPTIONAL:
            nargs_pattern = '(A?)' if option else '(-*A?-*)'

        # allow zero or more arguments
        elif nargs == ZERO_OR_MORE:
            nargs_pattern = '(A*)' if option else '(-*[A-]*)'

        # allow one or more arguments
        elif nargs == ONE_OR_MORE:
            nargs_pattern = '(A+)' if option else '(-*A[A-]*)'

        # allow any number of options or arguments
        elif nargs == REMAINDER:
            nargs_pattern = '([AO]*)' if option else '(.*)'

        # allow one argument followed by any number of options or arguments
        elif nargs == PARSER:
            nargs_pattern = '(A[AO]*)' if option else '(-*A[-AO]*)'

        # suppress action, like nargs=0
        elif nargs == SUPPRESS:
            nargs_pattern = '()' if option else '(-*)'

        # all others should be integers
        else:
            nargs_pattern = '([AO]{%d})' % nargs if option else '((?:-*A){%d}-*)' % nargs

        # return the pattern
        return nargs_pattern

    # ========================
    # Alt command line argument parsing, allowing free intermix
    # ========================

    def parse_intermixed_args(self, args=None, namespace=None):
        args, argv = self.parse_known_intermixed_args(args, namespace)
        if argv:
            msg = _('unrecognized arguments: %s') % ' '.join(argv)
            if self.exit_on_error:
                self.error(msg)
            else:
                raise ArgumentError(None, msg)
        return args

    def parse_known_intermixed_args(self, args=None, namespace=None):
        # returns a namespace and list of extras
        #
        # positional can be freely intermixed with optionals.  optionals are
        # first parsed with all positional arguments deactivated.  The 'extras'
        # are then parsed.  If the parser definition is incompatible with the
        # intermixed assumptions (e.g. use of REMAINDER, subparsers) a
        # TypeError is raised.

        positionals = self._get_positional_actions()
        a = [action for action in positionals
             if action.nargs in [PARSER, REMAINDER]]
        if a:
            raise TypeError('parse_intermixed_args: positional arg'
                            ' with nargs=%s'%a[0].nargs)

        return self._parse_known_args2(args, namespace, intermixed=True)

    # ========================
    # Value conversion methods
    # ========================

    def _get_values(self, action, arg_strings):
        # optional argument produces a default when not present
        if not arg_strings and action.nargs == OPTIONAL:
            if action.option_strings:
                value = action.const
            else:
                value = action.default
            if isinstance(value, str) and value is not SUPPRESS:
                value = self._get_value(action, value)
                self._check_value(action, value)

        # when nargs='*' on a positional, if there were no command-line
        # args, use the default if it is anything other than None
        elif (not arg_strings and action.nargs == ZERO_OR_MORE and
              not action.option_strings):
            if action.default is not None:
                value = action.default
                self._check_value(action, value)
            else:
                # since arg_strings is always [] at this point
                # there is no need to use self._check_value(action, value)
                value = arg_strings

        # single argument or optional argument produces a single value
        elif len(arg_strings) == 1 and action.nargs in [None, OPTIONAL]:
            arg_string, = arg_strings
            value = self._get_value(action, arg_string)
            self._check_value(action, value)

        # REMAINDER arguments convert all values, checking none
        elif action.nargs == REMAINDER:
            value = [self._get_value(action, v) for v in arg_strings]

        # PARSER arguments convert all values, but check only the first
        elif action.nargs == PARSER:
            value = [self._get_value(action, v) for v in arg_strings]
            self._check_value(action, value[0])

        # SUPPRESS argument does not put anything in the namespace
        elif action.nargs == SUPPRESS:
            value = SUPPRESS

        # all other types of nargs produce a list
        else:
            value = [self._get_value(action, v) for v in arg_strings]
            for v in value:
                self._check_value(action, v)

        # return the converted value
        return value

    def _get_value(self, action, arg_string):
        type_func = self._registry_get('type', action.type, action.type)
        if not callable(type_func):
            msg = _('%r is not callable')
            raise ArgumentError(action, msg % type_func)

        # convert the value to the appropriate type
        try:
            result = type_func(arg_string)

        # ArgumentTypeErrors indicate errors
        except ArgumentTypeError as err:
            msg = str(err)
            raise ArgumentError(action, msg)

        # TypeErrors or ValueErrors also indicate errors
        except (TypeError, ValueError):
            name = getattr(action.type, '__name__', repr(action.type))
            args = {'type': name, 'value': arg_string}
            msg = _('invalid %(type)s value: %(value)r')
            raise ArgumentError(action, msg % args)

        # return the converted value
        return result

    def _check_value(self, action, value):
        # converted value must be one of the choices (if specified)
        choices = action.choices
        if choices is not None:
            if isinstance(choices, str):
                choices = iter(choices)
            if value not in choices:
                args = {'value': str(value),
                        'choices': ', '.join(repr(str(choice)) for choice in action.choices)}
                msg = _('invalid choice: %(value)r (choose from %(choices)s)')
                raise ArgumentError(action, msg % args)

    # =======================
    # Help-formatting methods
    # =======================

    def format_usage(self):
        formatter = self._get_formatter()
        formatter.add_usage(self.usage, self._actions,
                            self._mutually_exclusive_groups)
        return formatter.format_help()

    def format_help(self):
        formatter = self._get_formatter()

        # usage
        formatter.add_usage(self.usage, self._actions,
                            self._mutually_exclusive_groups)

        # description
        formatter.add_text(self.description)

        # positionals, optionals and user-defined groups
        for action_group in self._action_groups:
            formatter.start_section(action_group.title)
            formatter.add_text(action_group.description)
            formatter.add_arguments(action_group._group_actions)
            formatter.end_section()

        # epilog
        formatter.add_text(self.epilog)

        # determine help from format above
        return formatter.format_help()

    def _get_formatter(self):
        return self.formatter_class(prog=self.prog)

    # =====================
    # Help-printing methods
    # =====================

    def print_usage(self, file=None):
        if file is None:
            file = _sys.stdout
        self._print_message(self.format_usage(), file)

    def print_help(self, file=None):
        if file is None:
            file = _sys.stdout
        self._print_message(self.format_help(), file)

    def _print_message(self, message, file=None):
        if message:
            file = file or _sys.stderr
            try:
                file.write(message)
            except (AttributeError, OSError):
                pass

    # ===============
    # Exiting methods
    # ===============

    def exit(self, status=0, message=None):
        if message:
            self._print_message(message, _sys.stderr)
        _sys.exit(status)

    def error(self, message):
        """error(message: string)

        Prints a usage message incorporating the message to stderr and
        exits.

        If you override this in a subclass, it should not return -- it
        should either exit or raise an exception.
        """
        self.print_usage(_sys.stderr)
        args = {'prog': self.prog, 'message': message}
        self.exit(2, _('%(prog)s: error: %(message)s\n') % args)

    def _warning(self, message):
        args = {'prog': self.prog, 'message': message}
        self._print_message(_('%(prog)s: warning: %(message)s\n') % args, _sys.stderr)



"""Implementation of JSONDecoder
"""
import re

from json import scanner
try:
    from _json import scanstring as c_scanstring
except ImportError:
    c_scanstring = None

__all__ = ['JSONDecoder', 'JSONDecodeError']

FLAGS = re.VERBOSE | re.MULTILINE | re.DOTALL

NaN = float('nan')
PosInf = float('inf')
NegInf = float('-inf')


class JSONDecodeError(ValueError):
    """Subclass of ValueError with the following additional properties:

    msg: The unformatted error message
    doc: The JSON document being parsed
    pos: The start index of doc where parsing failed
    lineno: The line corresponding to pos
    colno: The column corresponding to pos

    """
    # Note that this exception is used from _json
    def __init__(self, msg, doc, pos):
        lineno = doc.count('\n', 0, pos) + 1
        colno = pos - doc.rfind('\n', 0, pos)
        errmsg = '%s: line %d column %d (char %d)' % (msg, lineno, colno, pos)
        ValueError.__init__(self, errmsg)
        self.msg = msg
        self.doc = doc
        self.pos = pos
        self.lineno = lineno
        self.colno = colno

    def __reduce__(self):
        return self.__class__, (self.msg, self.doc, self.pos)


_CONSTANTS = {
    '-Infinity': NegInf,
    'Infinity': PosInf,
    'NaN': NaN,
}


HEXDIGITS = re.compile(r'[0-9A-Fa-f]{4}', FLAGS)
STRINGCHUNK = re.compile(r'(.*?)(["\\\x00-\x1f])', FLAGS)
BACKSLASH = {
    '"': '"', '\\': '\\', '/': '/',
    'b': '\b', 'f': '\f', 'n': '\n', 'r': '\r', 't': '\t',
}

def _decode_uXXXX(s, pos, _m=HEXDIGITS.match):
    esc = _m(s, pos + 1)
    if esc is not None:
        try:
            return int(esc.group(), 16)
        except ValueError:
            pass
    msg = "Invalid \\uXXXX escape"
    raise JSONDecodeError(msg, s, pos)

def py_scanstring(s, end, strict=True,
        _b=BACKSLASH, _m=STRINGCHUNK.match):
    """Scan the string s for a JSON string. End is the index of the
    character in s after the quote that started the JSON string.
    Unescapes all valid JSON string escape sequences and raises ValueError
    on attempt to decode an invalid string. If strict is False then literal
    control characters are allowed in the string.

    Returns a tuple of the decoded string and the index of the character in s
    after the end quote."""
    chunks = []
    _append = chunks.append
    begin = end - 1
    while 1:
        chunk = _m(s, end)
        if chunk is None:
            raise JSONDecodeError("Unterminated string starting at", s, begin)
        end = chunk.end()
        content, terminator = chunk.groups()
        # Content is contains zero or more unescaped string characters
        if content:
            _append(content)
        # Terminator is the end of string, a literal control character,
        # or a backslash denoting that an escape sequence follows
        if terminator == '"':
            break
        elif terminator != '\\':
            if strict:
                #msg = "Invalid control character %r at" % (terminator,)
                msg = "Invalid control character {0!r} at".format(terminator)
                raise JSONDecodeError(msg, s, end)
            else:
                _append(terminator)
                continue
        try:
            esc = s[end]
        except IndexError:
            raise JSONDecodeError("Unterminated string starting at",
                                  s, begin) from None
        # If not a unicode escape sequence, must be in the lookup table
        if esc != 'u':
            try:
                char = _b[esc]
            except KeyError:
                msg = "Invalid \\escape: {0!r}".format(esc)
                raise JSONDecodeError(msg, s, end)
            end += 1
        else:
            uni = _decode_uXXXX(s, end)
            end += 5
            if 0xd800 <= uni <= 0xdbff and s[end:end + 2] == '\\u':
                uni2 = _decode_uXXXX(s, end + 1)
                if 0xdc00 <= uni2 <= 0xdfff:
                    uni = 0x10000 + (((uni - 0xd800) << 10) | (uni2 - 0xdc00))
                    end += 6
            char = chr(uni)
        _append(char)
    return ''.join(chunks), end


# Use speedup if available
scanstring = c_scanstring or py_scanstring

WHITESPACE = re.compile(r'[ \t\n\r]*', FLAGS)
WHITESPACE_STR = ' \t\n\r'


def JSONObject(s_and_end, strict, scan_once, object_hook, object_pairs_hook,
               memo=None, _w=WHITESPACE.match, _ws=WHITESPACE_STR):
    s, end = s_and_end
    pairs = []
    pairs_append = pairs.append
    # Backwards compatibility
    if memo is None:
        memo = {}
    memo_get = memo.setdefault
    # Use a slice to prevent IndexError from being raised, the following
    # check will raise a more specific ValueError if the string is empty
    nextchar = s[end:end + 1]
    # Normally we expect nextchar == '"'
    if nextchar != '"':
        if nextchar in _ws:
            end = _w(s, end).end()
            nextchar = s[end:end + 1]
        # Trivial empty object
        if nextchar == '}':
            if object_pairs_hook is not None:
                result = object_pairs_hook(pairs)
                return result, end + 1
            pairs = {}
            if object_hook is not None:
                pairs = object_hook(pairs)
            return pairs, end + 1
        elif nextchar != '"':
            raise JSONDecodeError(
                "Expecting property name enclosed in double quotes", s, end)
    end += 1
    while True:
        key, end = scanstring(s, end, strict)
        key = memo_get(key, key)
        # To skip some function call overhead we optimize the fast paths where
        # the JSON key separator is ": " or just ":".
        if s[end:end + 1] != ':':
            end = _w(s, end).end()
            if s[end:end + 1] != ':':
                raise JSONDecodeError("Expecting ':' delimiter", s, end)
        end += 1

        try:
            if s[end] in _ws:
                end += 1
                if s[end] in _ws:
                    end = _w(s, end + 1).end()
        except IndexError:
            pass

        try:
            value, end = scan_once(s, end)
        except StopIteration as err:
            raise JSONDecodeError("Expecting value", s, err.value) from None
        pairs_append((key, value))
        try:
            nextchar = s[end]
            if nextchar in _ws:
                end = _w(s, end + 1).end()
                nextchar = s[end]
        except IndexError:
            nextchar = ''
        end += 1

        if nextchar == '}':
            break
        elif nextchar != ',':
            raise JSONDecodeError("Expecting ',' delimiter", s, end - 1)
        comma_idx = end - 1
        end = _w(s, end).end()
        nextchar = s[end:end + 1]
        end += 1
        if nextchar != '"':
            if nextchar == '}':
                raise JSONDecodeError("Illegal trailing comma before end of object", s, comma_idx)
            raise JSONDecodeError(
                "Expecting property name enclosed in double quotes", s, end - 1)
    if object_pairs_hook is not None:
        result = object_pairs_hook(pairs)
        return result, end
    pairs = dict(pairs)
    if object_hook is not None:
        pairs = object_hook(pairs)
    return pairs, end

def JSONArray(s_and_end, scan_once, _w=WHITESPACE.match, _ws=WHITESPACE_STR):
    s, end = s_and_end
    values = []
    nextchar = s[end:end + 1]
    if nextchar in _ws:
        end = _w(s, end + 1).end()
        nextchar = s[end:end + 1]
    # Look-ahead for trivial empty array
    if nextchar == ']':
        return values, end + 1
    _append = values.append
    while True:
        try:
            value, end = scan_once(s, end)
        except StopIteration as err:
            raise JSONDecodeError("Expecting value", s, err.value) from None
        _append(value)
        nextchar = s[end:end + 1]
        if nextchar in _ws:
            end = _w(s, end + 1).end()
            nextchar = s[end:end + 1]
        end += 1
        if nextchar == ']':
            break
        elif nextchar != ',':
            raise JSONDecodeError("Expecting ',' delimiter", s, end - 1)
        comma_idx = end - 1
        try:
            if s[end] in _ws:
                end += 1
                if s[end] in _ws:
                    end = _w(s, end + 1).end()
            nextchar = s[end:end + 1]
        except IndexError:
            pass
        if nextchar == ']':
            raise JSONDecodeError("Illegal trailing comma before end of array", s, comma_idx)

    return values, end


class JSONDecoder(object):
    """Simple JSON <https://json.org> decoder

    Performs the following translations in decoding by default:

    +---------------+-------------------+
    | JSON          | Python            |
    +===============+===================+
    | object        | dict              |
    +---------------+-------------------+
    | array         | list              |
    +---------------+-------------------+
    | string        | str               |
    +---------------+-------------------+
    | number (int)  | int               |
    +---------------+-------------------+
    | number (real) | float             |
    +---------------+-------------------+
    | true          | True              |
    +---------------+-------------------+
    | false         | False             |
    +---------------+-------------------+
    | null          | None              |
    +---------------+-------------------+

    It also understands ``NaN``, ``Infinity``, and ``-Infinity`` as
    their corresponding ``float`` values, which is outside the JSON spec.

    """

    def __init__(self, *, object_hook=None, parse_float=None,
            parse_int=None, parse_constant=None, strict=True,
            object_pairs_hook=None):
        """``object_hook``, if specified, will be called with the result
        of every JSON object decoded and its return value will be used in
        place of the given ``dict``.  This can be used to provide custom
        deserializations (e.g. to support JSON-RPC class hinting).

        ``object_pairs_hook``, if specified will be called with the result
        of every JSON object decoded with an ordered list of pairs.  The
        return value of ``object_pairs_hook`` will be used instead of the
        ``dict``.  This feature can be used to implement custom decoders.
        If ``object_hook`` is also defined, the ``object_pairs_hook`` takes
        priority.

        ``parse_float``, if specified, will be called with the string
        of every JSON float to be decoded. By default this is equivalent to
        float(num_str). This can be used to use another datatype or parser
        for JSON floats (e.g. decimal.Decimal).

        ``parse_int``, if specified, will be called with the string
        of every JSON int to be decoded. By default this is equivalent to
        int(num_str). This can be used to use another datatype or parser
        for JSON integers (e.g. float).

        ``parse_constant``, if specified, will be called with one of the
        following strings: -Infinity, Infinity, NaN.
        This can be used to raise an exception if invalid JSON numbers
        are encountered.

        If ``strict`` is false (true is the default), then control
        characters will be allowed inside strings.  Control characters in
        this context are those with character codes in the 0-31 range,
        including ``'\\t'`` (tab), ``'\\n'``, ``'\\r'`` and ``'\\0'``.
        """
        self.object_hook = object_hook
        self.parse_float = parse_float or float
        self.parse_int = parse_int or int
        self.parse_constant = parse_constant or _CONSTANTS.__getitem__
        self.strict = strict
        self.object_pairs_hook = object_pairs_hook
        self.parse_object = JSONObject
        self.parse_array = JSONArray
        self.parse_string = scanstring
        self.memo = {}
        self.scan_once = scanner.make_scanner(self)


    def decode(self, s, _w=WHITESPACE.match):
        """Return the Python representation of ``s`` (a ``str`` instance
        containing a JSON document).

        """
        obj, end = self.raw_decode(s, idx=_w(s, 0).end())
        end = _w(s, end).end()
        if end != len(s):
            raise JSONDecodeError("Extra data", s, end)
        return obj

    def raw_decode(self, s, idx=0):
        """Decode a JSON document from ``s`` (a ``str`` beginning with
        a JSON document) and return a 2-tuple of the Python
        representation and the index in ``s`` where the document ended.

        This can be used to decode a JSON document from a string that may
        have extraneous data at the end.

        """
        try:
            obj, end = self.scan_once(s, idx)
        except StopIteration as err:
            raise JSONDecodeError("Expecting value", s, err.value) from None
        return obj, end



"""Implementation of JSONEncoder
"""
import re

try:
    from _json import encode_basestring_ascii as c_encode_basestring_ascii
except ImportError:
    c_encode_basestring_ascii = None
try:
    from _json import encode_basestring as c_encode_basestring
except ImportError:
    c_encode_basestring = None
try:
    from _json import make_encoder as c_make_encoder
except ImportError:
    c_make_encoder = None

ESCAPE = re.compile(r'[\x00-\x1f\\"\b\f\n\r\t]')
ESCAPE_ASCII = re.compile(r'([\\"]|[^\ -~])')
HAS_UTF8 = re.compile(b'[\x80-\xff]')
ESCAPE_DCT = {
    '\\': '\\\\',
    '"': '\\"',
    '\b': '\\b',
    '\f': '\\f',
    '\n': '\\n',
    '\r': '\\r',
    '\t': '\\t',
}
for i in range(0x20):
    ESCAPE_DCT.setdefault(chr(i), '\\u{0:04x}'.format(i))
    #ESCAPE_DCT.setdefault(chr(i), '\\u%04x' % (i,))
del i

INFINITY = float('inf')

def py_encode_basestring(s):
    """Return a JSON representation of a Python string

    """
    def replace(match):
        return ESCAPE_DCT[match.group(0)]
    return '"' + ESCAPE.sub(replace, s) + '"'


encode_basestring = (c_encode_basestring or py_encode_basestring)


def py_encode_basestring_ascii(s):
    """Return an ASCII-only JSON representation of a Python string

    """
    def replace(match):
        s = match.group(0)
        try:
            return ESCAPE_DCT[s]
        except KeyError:
            n = ord(s)
            if n < 0x10000:
                return '\\u{0:04x}'.format(n)
                #return '\\u%04x' % (n,)
            else:
                # surrogate pair
                n -= 0x10000
                s1 = 0xd800 | ((n >> 10) & 0x3ff)
                s2 = 0xdc00 | (n & 0x3ff)
                return '\\u{0:04x}\\u{1:04x}'.format(s1, s2)
    return '"' + ESCAPE_ASCII.sub(replace, s) + '"'


encode_basestring_ascii = (
    c_encode_basestring_ascii or py_encode_basestring_ascii)

class JSONEncoder(object):
    """Extensible JSON <https://json.org> encoder for Python data structures.

    Supports the following objects and types by default:

    +-------------------+---------------+
    | Python            | JSON          |
    +===================+===============+
    | dict              | object        |
    +-------------------+---------------+
    | list, tuple       | array         |
    +-------------------+---------------+
    | str               | string        |
    +-------------------+---------------+
    | int, float        | number        |
    +-------------------+---------------+
    | True              | true          |
    +-------------------+---------------+
    | False             | false         |
    +-------------------+---------------+
    | None              | null          |
    +-------------------+---------------+

    To extend this to recognize other objects, subclass and implement a
    ``.default()`` method with another method that returns a serializable
    object for ``o`` if possible, otherwise it should call the superclass
    implementation (to raise ``TypeError``).

    """
    item_separator = ', '
    key_separator = ': '
    def __init__(self, *, skipkeys=False, ensure_ascii=True,
            check_circular=True, allow_nan=True, sort_keys=False,
            indent=None, separators=None, default=None):
        """Constructor for JSONEncoder, with sensible defaults.

        If skipkeys is false, then it is a TypeError to attempt
        encoding of keys that are not str, int, float, bool or None.
        If skipkeys is True, such items are simply skipped.

        If ensure_ascii is true, the output is guaranteed to be str objects
        with all incoming non-ASCII and non-printable characters escaped.
        If ensure_ascii is false, the output can contain non-ASCII and
        non-printable characters.

        If check_circular is true, then lists, dicts, and custom encoded
        objects will be checked for circular references during encoding to
        prevent an infinite recursion (which would cause an RecursionError).
        Otherwise, no such check takes place.

        If allow_nan is true, then NaN, Infinity, and -Infinity will be
        encoded as such.  This behavior is not JSON specification compliant,
        but is consistent with most JavaScript based encoders and decoders.
        Otherwise, it will be a ValueError to encode such floats.

        If sort_keys is true, then the output of dictionaries will be
        sorted by key; this is useful for regression tests to ensure
        that JSON serializations can be compared on a day-to-day basis.

        If indent is a non-negative integer, then JSON array
        elements and object members will be pretty-printed with that
        indent level.  An indent level of 0 will only insert newlines.
        None is the most compact representation.

        If specified, separators should be an (item_separator,
        key_separator) tuple.  The default is (', ', ': ') if *indent* is
        ``None`` and (',', ': ') otherwise.  To get the most compact JSON
        representation, you should specify (',', ':') to eliminate
        whitespace.

        If specified, default is a function that gets called for objects
        that can't otherwise be serialized.  It should return a JSON
        encodable version of the object or raise a ``TypeError``.

        """

        self.skipkeys = skipkeys
        self.ensure_ascii = ensure_ascii
        self.check_circular = check_circular
        self.allow_nan = allow_nan
        self.sort_keys = sort_keys
        self.indent = indent
        if separators is not None:
            self.item_separator, self.key_separator = separators
        elif indent is not None:
            self.item_separator = ','
        if default is not None:
            self.default = default

    def default(self, o):
        """Implement this method in a subclass such that it returns
        a serializable object for ``o``, or calls the base implementation
        (to raise a ``TypeError``).

        For example, to support arbitrary iterators, you could
        implement default like this::

            def default(self, o):
                try:
                    iterable = iter(o)
                except TypeError:
                    pass
                else:
                    return list(iterable)
                # Let the base class default method raise the TypeError
                return super().default(o)

        """
        raise TypeError(f'Object of type {o.__class__.__name__} '
                        f'is not JSON serializable')

    def encode(self, o):
        """Return a JSON string representation of a Python data structure.

        >>> from json.encoder import JSONEncoder
        >>> JSONEncoder().encode({"foo": ["bar", "baz"]})
        '{"foo": ["bar", "baz"]}'

        """
        # This is for extremely simple cases and benchmarks.
        if isinstance(o, str):
            if self.ensure_ascii:
                return encode_basestring_ascii(o)
            else:
                return encode_basestring(o)
        # This doesn't pass the iterator directly to ''.join() because the
        # exceptions aren't as detailed.  The list call should be roughly
        # equivalent to the PySequence_Fast that ''.join() would do.
        chunks = self.iterencode(o, _one_shot=True)
        if not isinstance(chunks, (list, tuple)):
            chunks = list(chunks)
        return ''.join(chunks)

    def iterencode(self, o, _one_shot=False):
        """Encode the given object and yield each string
        representation as available.

        For example::

            for chunk in JSONEncoder().iterencode(bigobject):
                mysocket.write(chunk)

        """
        if self.check_circular:
            markers = {}
        else:
            markers = None
        if self.ensure_ascii:
            _encoder = encode_basestring_ascii
        else:
            _encoder = encode_basestring

        def floatstr(o, allow_nan=self.allow_nan,
                _repr=float.__repr__, _inf=INFINITY, _neginf=-INFINITY):
            # Check for specials.  Note that this type of test is processor
            # and/or platform-specific, so do tests which don't depend on the
            # internals.

            if o != o:
                text = 'NaN'
            elif o == _inf:
                text = 'Infinity'
            elif o == _neginf:
                text = '-Infinity'
            else:
                return _repr(o)

            if not allow_nan:
                raise ValueError(
                    "Out of range float values are not JSON compliant: " +
                    repr(o))

            return text


        if self.indent is None or isinstance(self.indent, str):
            indent = self.indent
        else:
            indent = ' ' * self.indent
        if _one_shot and c_make_encoder is not None:
            _iterencode = c_make_encoder(
                markers, self.default, _encoder, indent,
                self.key_separator, self.item_separator, self.sort_keys,
                self.skipkeys, self.allow_nan)
        else:
            _iterencode = _make_iterencode(
                markers, self.default, _encoder, indent, floatstr,
                self.key_separator, self.item_separator, self.sort_keys,
                self.skipkeys, _one_shot)
        return _iterencode(o, 0)

def _make_iterencode(markers, _default, _encoder, _indent, _floatstr,
        _key_separator, _item_separator, _sort_keys, _skipkeys, _one_shot,
        ## HACK: hand-optimized bytecode; turn globals into locals
        ValueError=ValueError,
        dict=dict,
        float=float,
        id=id,
        int=int,
        isinstance=isinstance,
        list=list,
        str=str,
        tuple=tuple,
        _intstr=int.__repr__,
    ):

    def _iterencode_list(lst, _current_indent_level):
        if not lst:
            yield '[]'
            return
        if markers is not None:
            markerid = id(lst)
            if markerid in markers:
                raise ValueError("Circular reference detected")
            markers[markerid] = lst
        buf = '['
        if _indent is not None:
            _current_indent_level += 1
            newline_indent = '\n' + _indent * _current_indent_level
            separator = _item_separator + newline_indent
            buf += newline_indent
        else:
            newline_indent = None
            separator = _item_separator
        first = True
        for value in lst:
            if first:
                first = False
            else:
                buf = separator
            if isinstance(value, str):
                yield buf + _encoder(value)
            elif value is None:
                yield buf + 'null'
            elif value is True:
                yield buf + 'true'
            elif value is False:
                yield buf + 'false'
            elif isinstance(value, int):
                # Subclasses of int/float may override __repr__, but we still
                # want to encode them as integers/floats in JSON. One example
                # within the standard library is IntEnum.
                yield buf + _intstr(value)
            elif isinstance(value, float):
                # see comment above for int
                yield buf + _floatstr(value)
            else:
                yield buf
                if isinstance(value, (list, tuple)):
                    chunks = _iterencode_list(value, _current_indent_level)
                elif isinstance(value, dict):
                    chunks = _iterencode_dict(value, _current_indent_level)
                else:
                    chunks = _iterencode(value, _current_indent_level)
                yield from chunks
        if newline_indent is not None:
            _current_indent_level -= 1
            yield '\n' + _indent * _current_indent_level
        yield ']'
        if markers is not None:
            del markers[markerid]

    def _iterencode_dict(dct, _current_indent_level):
        if not dct:
            yield '{}'
            return
        if markers is not None:
            markerid = id(dct)
            if markerid in markers:
                raise ValueError("Circular reference detected")
            markers[markerid] = dct
        yield '{'
        if _indent is not None:
            _current_indent_level += 1
            newline_indent = '\n' + _indent * _current_indent_level
            item_separator = _item_separator + newline_indent
        else:
            newline_indent = None
            item_separator = _item_separator
        first = True
        if _sort_keys:
            items = sorted(dct.items())
        else:
            items = dct.items()
        for key, value in items:
            if isinstance(key, str):
                pass
            # JavaScript is weakly typed for these, so it makes sense to
            # also allow them.  Many encoders seem to do something like this.
            elif isinstance(key, float):
                # see comment for int/float in _make_iterencode
                key = _floatstr(key)
            elif key is True:
                key = 'true'
            elif key is False:
                key = 'false'
            elif key is None:
                key = 'null'
            elif isinstance(key, int):
                # see comment for int/float in _make_iterencode
                key = _intstr(key)
            elif _skipkeys:
                continue
            else:
                raise TypeError(f'keys must be str, int, float, bool or None, '
                                f'not {key.__class__.__name__}')
            if first:
                first = False
                if newline_indent is not None:
                    yield newline_indent
            else:
                yield item_separator
            yield _encoder(key)
            yield _key_separator
            if isinstance(value, str):
                yield _encoder(value)
            elif value is None:
                yield 'null'
            elif value is True:
                yield 'true'
            elif value is False:
                yield 'false'
            elif isinstance(value, int):
                # see comment for int/float in _make_iterencode
                yield _intstr(value)
            elif isinstance(value, float):
                # see comment for int/float in _make_iterencode
                yield _floatstr(value)
            else:
                if isinstance(value, (list, tuple)):
                    chunks = _iterencode_list(value, _current_indent_level)
                elif isinstance(value, dict):
                    chunks = _iterencode_dict(value, _current_indent_level)
                else:
                    chunks = _iterencode(value, _current_indent_level)
                yield from chunks
        if not first and newline_indent is not None:
            _current_indent_level -= 1
            yield '\n' + _indent * _current_indent_level
        yield '}'
        if markers is not None:
            del markers[markerid]

    def _iterencode(o, _current_indent_level):
        if isinstance(o, str):
            yield _encoder(o)
        elif o is None:
            yield 'null'
        elif o is True:
            yield 'true'
        elif o is False:
            yield 'false'
        elif isinstance(o, int):
            # see comment for int/float in _make_iterencode
            yield _intstr(o)
        elif isinstance(o, float):
            # see comment for int/float in _make_iterencode
            yield _floatstr(o)
        elif isinstance(o, (list, tuple)):
            yield from _iterencode_list(o, _current_indent_level)
        elif isinstance(o, dict):
            yield from _iterencode_dict(o, _current_indent_level)
        else:
            if markers is not None:
                markerid = id(o)
                if markerid in markers:
                    raise ValueError("Circular reference detected")
                markers[markerid] = o
            o = _default(o)
            yield from _iterencode(o, _current_indent_level)
            if markers is not None:
                del markers[markerid]
    return _iterencode



import re
import sys
import copy
import types
import inspect
import keyword
import itertools
import abc
from reprlib import recursive_repr


__all__ = ['dataclass',
           'field',
           'Field',
           'FrozenInstanceError',
           'InitVar',
           'KW_ONLY',
           'MISSING',

           # Helper functions.
           'fields',
           'asdict',
           'astuple',
           'make_dataclass',
           'replace',
           'is_dataclass',
           ]

# Conditions for adding methods.  The boxes indicate what action the
# dataclass decorator takes.  For all of these tables, when I talk
# about init=, repr=, eq=, order=, unsafe_hash=, or frozen=, I'm
# referring to the arguments to the @dataclass decorator.  When
# checking if a dunder method already exists, I mean check for an
# entry in the class's __dict__.  I never check to see if an attribute
# is defined in a base class.

# Key:
# +=========+=========================================+
# + Value   | Meaning                                 |
# +=========+=========================================+
# | <blank> | No action: no method is added.          |
# +---------+-----------------------------------------+
# | add     | Generated method is added.              |
# +---------+-----------------------------------------+
# | raise   | TypeError is raised.                    |
# +---------+-----------------------------------------+
# | None    | Attribute is set to None.               |
# +=========+=========================================+

# __init__
#
#   +--- init= parameter
#   |
#   v     |       |       |
#         |  no   |  yes  |  <--- class has __init__ in __dict__?
# +=======+=======+=======+
# | False |       |       |
# +-------+-------+-------+
# | True  | add   |       |  <- the default
# +=======+=======+=======+

# __repr__
#
#    +--- repr= parameter
#    |
#    v    |       |       |
#         |  no   |  yes  |  <--- class has __repr__ in __dict__?
# +=======+=======+=======+
# | False |       |       |
# +-------+-------+-------+
# | True  | add   |       |  <- the default
# +=======+=======+=======+


# __setattr__
# __delattr__
#
#    +--- frozen= parameter
#    |
#    v    |       |       |
#         |  no   |  yes  |  <--- class has __setattr__ or __delattr__ in __dict__?
# +=======+=======+=======+
# | False |       |       |  <- the default
# +-------+-------+-------+
# | True  | add   | raise |
# +=======+=======+=======+
# Raise because not adding these methods would break the "frozen-ness"
# of the class.

# __eq__
#
#    +--- eq= parameter
#    |
#    v    |       |       |
#         |  no   |  yes  |  <--- class has __eq__ in __dict__?
# +=======+=======+=======+
# | False |       |       |
# +-------+-------+-------+
# | True  | add   |       |  <- the default
# +=======+=======+=======+

# __lt__
# __le__
# __gt__
# __ge__
#
#    +--- order= parameter
#    |
#    v    |       |       |
#         |  no   |  yes  |  <--- class has any comparison method in __dict__?
# +=======+=======+=======+
# | False |       |       |  <- the default
# +-------+-------+-------+
# | True  | add   | raise |
# +=======+=======+=======+
# Raise because to allow this case would interfere with using
# functools.total_ordering.

# __hash__

#    +------------------- unsafe_hash= parameter
#    |       +----------- eq= parameter
#    |       |       +--- frozen= parameter
#    |       |       |
#    v       v       v    |        |        |
#                         |   no   |  yes   |  <--- class has explicitly defined __hash__
# +=======+=======+=======+========+========+
# | False | False | False |        |        | No __eq__, use the base class __hash__
# +-------+-------+-------+--------+--------+
# | False | False | True  |        |        | No __eq__, use the base class __hash__
# +-------+-------+-------+--------+--------+
# | False | True  | False | None   |        | <-- the default, not hashable
# +-------+-------+-------+--------+--------+
# | False | True  | True  | add    |        | Frozen, so hashable, allows override
# +-------+-------+-------+--------+--------+
# | True  | False | False | add    | raise  | Has no __eq__, but hashable
# +-------+-------+-------+--------+--------+
# | True  | False | True  | add    | raise  | Has no __eq__, but hashable
# +-------+-------+-------+--------+--------+
# | True  | True  | False | add    | raise  | Not frozen, but hashable
# +-------+-------+-------+--------+--------+
# | True  | True  | True  | add    | raise  | Frozen, so hashable
# +=======+=======+=======+========+========+
# For boxes that are blank, __hash__ is untouched and therefore
# inherited from the base class.  If the base is object, then
# id-based hashing is used.
#
# Note that a class may already have __hash__=None if it specified an
# __eq__ method in the class body (not one that was created by
# @dataclass).
#
# See _hash_action (below) for a coded version of this table.

# __match_args__
#
#    +--- match_args= parameter
#    |
#    v    |       |       |
#         |  no   |  yes  |  <--- class has __match_args__ in __dict__?
# +=======+=======+=======+
# | False |       |       |
# +-------+-------+-------+
# | True  | add   |       |  <- the default
# +=======+=======+=======+
# __match_args__ is always added unless the class already defines it. It is a
# tuple of __init__ parameter names; non-init fields must be matched by keyword.


# Raised when an attempt is made to modify a frozen class.
class FrozenInstanceError(AttributeError): pass

# A sentinel object for default values to signal that a default
# factory will be used.  This is given a nice repr() which will appear
# in the function signature of dataclasses' constructors.
class _HAS_DEFAULT_FACTORY_CLASS:
    def __repr__(self):
        return '<factory>'
_HAS_DEFAULT_FACTORY = _HAS_DEFAULT_FACTORY_CLASS()

# A sentinel object to detect if a parameter is supplied or not.  Use
# a class to give it a better repr.
class _MISSING_TYPE:
    pass
MISSING = _MISSING_TYPE()

# A sentinel object to indicate that following fields are keyword-only by
# default.  Use a class to give it a better repr.
class _KW_ONLY_TYPE:
    pass
KW_ONLY = _KW_ONLY_TYPE()

# Since most per-field metadata will be unused, create an empty
# read-only proxy that can be shared among all fields.
_EMPTY_METADATA = types.MappingProxyType({})

# Markers for the various kinds of fields and pseudo-fields.
class _FIELD_BASE:
    def __init__(self, name):
        self.name = name
    def __repr__(self):
        return self.name
_FIELD = _FIELD_BASE('_FIELD')
_FIELD_CLASSVAR = _FIELD_BASE('_FIELD_CLASSVAR')
_FIELD_INITVAR = _FIELD_BASE('_FIELD_INITVAR')

# The name of an attribute on the class where we store the Field
# objects.  Also used to check if a class is a Data Class.
_FIELDS = '__dataclass_fields__'

# The name of an attribute on the class that stores the parameters to
# @dataclass.
_PARAMS = '__dataclass_params__'

# The name of the function, that if it exists, is called at the end of
# __init__.
_POST_INIT_NAME = '__post_init__'

# String regex that string annotations for ClassVar or InitVar must match.
# Allows "identifier.identifier[" or "identifier[".
# https://bugs.python.org/issue33453 for details.
_MODULE_IDENTIFIER_RE = re.compile(r'^(?:\s*(\w+)\s*\.)?\s*(\w+)')

# Atomic immutable types which don't require any recursive handling and for which deepcopy
# returns the same object. We can provide a fast-path for these types in asdict and astuple.
_ATOMIC_TYPES = frozenset({
    # Common JSON Serializable types
    types.NoneType,
    bool,
    int,
    float,
    str,
    # Other common types
    complex,
    bytes,
    # Other types that are also unaffected by deepcopy
    types.EllipsisType,
    types.NotImplementedType,
    types.CodeType,
    types.BuiltinFunctionType,
    types.FunctionType,
    type,
    range,
    property,
})


class InitVar:
    __slots__ = ('type', )

    def __init__(self, type):
        self.type = type

    def __repr__(self):
        if isinstance(self.type, type):
            type_name = self.type.__name__
        else:
            # typing objects, e.g. List[int]
            type_name = repr(self.type)
        return f'dataclasses.InitVar[{type_name}]'

    def __class_getitem__(cls, type):
        return InitVar(type)

# Instances of Field are only ever created from within this module,
# and only from the field() function, although Field instances are
# exposed externally as (conceptually) read-only objects.
#
# name and type are filled in after the fact, not in __init__.
# They're not known at the time this class is instantiated, but it's
# convenient if they're available later.
#
# When cls._FIELDS is filled in with a list of Field objects, the name
# and type fields will have been populated.
class Field:
    __slots__ = ('name',
                 'type',
                 'default',
                 'default_factory',
                 'repr',
                 'hash',
                 'init',
                 'compare',
                 'metadata',
                 'kw_only',
                 '_field_type',  # Private: not to be used by user code.
                 )

    def __init__(self, default, default_factory, init, repr, hash, compare,
                 metadata, kw_only):
        self.name = None
        self.type = None
        self.default = default
        self.default_factory = default_factory
        self.init = init
        self.repr = repr
        self.hash = hash
        self.compare = compare
        self.metadata = (_EMPTY_METADATA
                         if metadata is None else
                         types.MappingProxyType(metadata))
        self.kw_only = kw_only
        self._field_type = None

    @recursive_repr()
    def __repr__(self):
        return ('Field('
                f'name={self.name!r},'
                f'type={self.type!r},'
                f'default={self.default!r},'
                f'default_factory={self.default_factory!r},'
                f'init={self.init!r},'
                f'repr={self.repr!r},'
                f'hash={self.hash!r},'
                f'compare={self.compare!r},'
                f'metadata={self.metadata!r},'
                f'kw_only={self.kw_only!r},'
                f'_field_type={self._field_type}'
                ')')

    # This is used to support the PEP 487 __set_name__ protocol in the
    # case where we're using a field that contains a descriptor as a
    # default value.  For details on __set_name__, see
    # https://peps.python.org/pep-0487/#implementation-details.
    #
    # Note that in _process_class, this Field object is overwritten
    # with the default value, so the end result is a descriptor that
    # had __set_name__ called on it at the right time.
    def __set_name__(self, owner, name):
        func = getattr(type(self.default), '__set_name__', None)
        if func:
            # There is a __set_name__ method on the descriptor, call
            # it.
            func(self.default, owner, name)

    __class_getitem__ = classmethod(types.GenericAlias)


class _DataclassParams:
    __slots__ = ('init',
                 'repr',
                 'eq',
                 'order',
                 'unsafe_hash',
                 'frozen',
                 'match_args',
                 'kw_only',
                 'slots',
                 'weakref_slot',
                 )

    def __init__(self,
                 init, repr, eq, order, unsafe_hash, frozen,
                 match_args, kw_only, slots, weakref_slot):
        self.init = init
        self.repr = repr
        self.eq = eq
        self.order = order
        self.unsafe_hash = unsafe_hash
        self.frozen = frozen
        self.match_args = match_args
        self.kw_only = kw_only
        self.slots = slots
        self.weakref_slot = weakref_slot

    def __repr__(self):
        return ('_DataclassParams('
                f'init={self.init!r},'
                f'repr={self.repr!r},'
                f'eq={self.eq!r},'
                f'order={self.order!r},'
                f'unsafe_hash={self.unsafe_hash!r},'
                f'frozen={self.frozen!r},'
                f'match_args={self.match_args!r},'
                f'kw_only={self.kw_only!r},'
                f'slots={self.slots!r},'
                f'weakref_slot={self.weakref_slot!r}'
                ')')


# This function is used instead of exposing Field creation directly,
# so that a type checker can be told (via overloads) that this is a
# function whose type depends on its parameters.
def field(*, default=MISSING, default_factory=MISSING, init=True, repr=True,
          hash=None, compare=True, metadata=None, kw_only=MISSING):
    """Return an object to identify dataclass fields.

    default is the default value of the field.  default_factory is a
    0-argument function called to initialize a field's value.  If init
    is true, the field will be a parameter to the class's __init__()
    function.  If repr is true, the field will be included in the
    object's repr().  If hash is true, the field will be included in the
    object's hash().  If compare is true, the field will be used in
    comparison functions.  metadata, if specified, must be a mapping
    which is stored but not otherwise examined by dataclass.  If kw_only
    is true, the field will become a keyword-only parameter to
    __init__().

    It is an error to specify both default and default_factory.
    """

    if default is not MISSING and default_factory is not MISSING:
        raise ValueError('cannot specify both default and default_factory')
    return Field(default, default_factory, init, repr, hash, compare,
                 metadata, kw_only)


def _fields_in_init_order(fields):
    # Returns the fields as __init__ will output them.  It returns 2 tuples:
    # the first for normal args, and the second for keyword args.

    return (tuple(f for f in fields if f.init and not f.kw_only),
            tuple(f for f in fields if f.init and f.kw_only)
            )


def _tuple_str(obj_name, fields):
    # Return a string representing each field of obj_name as a tuple
    # member.  So, if fields is ['x', 'y'] and obj_name is "self",
    # return "(self.x,self.y)".

    # Special case for the 0-tuple.
    if not fields:
        return '()'
    # Note the trailing comma, needed if this turns out to be a 1-tuple.
    return f'({",".join([f"{obj_name}.{f.name}" for f in fields])},)'


class _FuncBuilder:
    def __init__(self, globals):
        self.names = []
        self.src = []
        self.globals = globals
        self.locals = {}
        self.overwrite_errors = {}
        self.unconditional_adds = {}

    def add_fn(self, name, args, body, *, locals=None, return_type=MISSING,
               overwrite_error=False, unconditional_add=False, decorator=None):
        if locals is not None:
            self.locals.update(locals)

        # Keep track if this method is allowed to be overwritten if it already
        # exists in the class.  The error is method-specific, so keep it with
        # the name.  We'll use this when we generate all of the functions in
        # the add_fns_to_class call.  overwrite_error is either True, in which
        # case we'll raise an error, or it's a string, in which case we'll
        # raise an error and append this string.
        if overwrite_error:
            self.overwrite_errors[name] = overwrite_error

        # Should this function always overwrite anything that's already in the
        # class?  The default is to not overwrite a function that already
        # exists.
        if unconditional_add:
            self.unconditional_adds[name] = True

        self.names.append(name)

        if return_type is not MISSING:
            self.locals[f'__dataclass_{name}_return_type__'] = return_type
            return_annotation = f'->__dataclass_{name}_return_type__'
        else:
            return_annotation = ''
        args = ','.join(args)
        body = '\n'.join(body)

        # Compute the text of the entire function, add it to the text we're generating.
        self.src.append(f'{f' {decorator}\n' if decorator else ''} def {name}({args}){return_annotation}:\n{body}')

    def add_fns_to_class(self, cls):
        # The source to all of the functions we're generating.
        fns_src = '\n'.join(self.src)

        # The locals they use.
        local_vars = ','.join(self.locals.keys())

        # The names of all of the functions, used for the return value of the
        # outer function.  Need to handle the 0-tuple specially.
        if len(self.names) == 0:
            return_names = '()'
        else:
            return_names  =f'({",".join(self.names)},)'

        # txt is the entire function we're going to execute, including the
        # bodies of the functions we're defining.  Here's a greatly simplified
        # version:
        # def __create_fn__():
        #  def __init__(self, x, y):
        #   self.x = x
        #   self.y = y
        #  @recursive_repr
        #  def __repr__(self):
        #   return f"cls(x={self.x!r},y={self.y!r})"
        # return __init__,__repr__

        txt = f"def __create_fn__({local_vars}):\n{fns_src}\n return {return_names}"
        ns = {}
        exec(txt, self.globals, ns)
        fns = ns['__create_fn__'](**self.locals)

        # Now that we've generated the functions, assign them into cls.
        for name, fn in zip(self.names, fns):
            fn.__qualname__ = f"{cls.__qualname__}.{fn.__name__}"
            if self.unconditional_adds.get(name, False):
                setattr(cls, name, fn)
            else:
                already_exists = _set_new_attribute(cls, name, fn)

                # See if it's an error to overwrite this particular function.
                if already_exists and (msg_extra := self.overwrite_errors.get(name)):
                    error_msg = (f'Cannot overwrite attribute {fn.__name__} '
                                 f'in class {cls.__name__}')
                    if not msg_extra is True:
                        error_msg = f'{error_msg} {msg_extra}'

                    raise TypeError(error_msg)


def _field_assign(frozen, name, value, self_name):
    # If we're a frozen class, then assign to our fields in __init__
    # via object.__setattr__.  Otherwise, just use a simple
    # assignment.
    #
    # self_name is what "self" is called in this function: don't
    # hard-code "self", since that might be a field name.
    if frozen:
        return f'  __dataclass_builtins_object__.__setattr__({self_name},{name!r},{value})'
    return f'  {self_name}.{name}={value}'


def _field_init(f, frozen, globals, self_name, slots):
    # Return the text of the line in the body of __init__ that will
    # initialize this field.

    default_name = f'__dataclass_dflt_{f.name}__'
    if f.default_factory is not MISSING:
        if f.init:
            # This field has a default factory.  If a parameter is
            # given, use it.  If not, call the factory.
            globals[default_name] = f.default_factory
            value = (f'{default_name}() '
                     f'if {f.name} is __dataclass_HAS_DEFAULT_FACTORY__ '
                     f'else {f.name}')
        else:
            # This is a field that's not in the __init__ params, but
            # has a default factory function.  It needs to be
            # initialized here by calling the factory function,
            # because there's no other way to initialize it.

            # For a field initialized with a default=defaultvalue, the
            # class dict just has the default value
            # (cls.fieldname=defaultvalue).  But that won't work for a
            # default factory, the factory must be called in __init__
            # and we must assign that to self.fieldname.  We can't
            # fall back to the class dict's value, both because it's
            # not set, and because it might be different per-class
            # (which, after all, is why we have a factory function!).

            globals[default_name] = f.default_factory
            value = f'{default_name}()'
    else:
        # No default factory.
        if f.init:
            if f.default is MISSING:
                # There's no default, just do an assignment.
                value = f.name
            elif f.default is not MISSING:
                globals[default_name] = f.default
                value = f.name
        else:
            # If the class has slots, then initialize this field.
            if slots and f.default is not MISSING:
                globals[default_name] = f.default
                value = default_name
            else:
                # This field does not need initialization: reading from it will
                # just use the class attribute that contains the default.
                # Signify that to the caller by returning None.
                return None

    # Only test this now, so that we can create variables for the
    # default.  However, return None to signify that we're not going
    # to actually do the assignment statement for InitVars.
    if f._field_type is _FIELD_INITVAR:
        return None

    # Now, actually generate the field assignment.
    return _field_assign(frozen, f.name, value, self_name)


def _init_param(f):
    # Return the __init__ parameter string for this field.  For
    # example, the equivalent of 'x:int=3' (except instead of 'int',
    # reference a variable set to int, and instead of '3', reference a
    # variable set to 3).
    if f.default is MISSING and f.default_factory is MISSING:
        # There's no default, and no default_factory, just output the
        # variable name and type.
        default = ''
    elif f.default is not MISSING:
        # There's a default, this will be the name that's used to look
        # it up.
        default = f'=__dataclass_dflt_{f.name}__'
    elif f.default_factory is not MISSING:
        # There's a factory function.  Set a marker.
        default = '=__dataclass_HAS_DEFAULT_FACTORY__'
    return f'{f.name}:__dataclass_type_{f.name}__{default}'


def _init_fn(fields, std_fields, kw_only_fields, frozen, has_post_init,
             self_name, func_builder, slots):
    # fields contains both real fields and InitVar pseudo-fields.

    # Make sure we don't have fields without defaults following fields
    # with defaults.  This actually would be caught when exec-ing the
    # function source code, but catching it here gives a better error
    # message, and future-proofs us in case we build up the function
    # using ast.

    seen_default = None
    for f in std_fields:
        # Only consider the non-kw-only fields in the __init__ call.
        if f.init:
            if not (f.default is MISSING and f.default_factory is MISSING):
                seen_default = f
            elif seen_default:
                raise TypeError(f'non-default argument {f.name!r} '
                                f'follows default argument {seen_default.name!r}')

    locals = {**{f'__dataclass_type_{f.name}__': f.type for f in fields},
              **{'__dataclass_HAS_DEFAULT_FACTORY__': _HAS_DEFAULT_FACTORY,
                 '__dataclass_builtins_object__': object,
                 }
              }

    body_lines = []
    for f in fields:
        line = _field_init(f, frozen, locals, self_name, slots)
        # line is None means that this field doesn't require
        # initialization (it's a pseudo-field).  Just skip it.
        if line:
            body_lines.append(line)

    # Does this class have a post-init function?
    if has_post_init:
        params_str = ','.join(f.name for f in fields
                              if f._field_type is _FIELD_INITVAR)
        body_lines.append(f'  {self_name}.{_POST_INIT_NAME}({params_str})')

    # If no body lines, use 'pass'.
    if not body_lines:
        body_lines = ['  pass']

    _init_params = [_init_param(f) for f in std_fields]
    if kw_only_fields:
        # Add the keyword-only args.  Because the * can only be added if
        # there's at least one keyword-only arg, there needs to be a test here
        # (instead of just concatenting the lists together).
        _init_params += ['*']
        _init_params += [_init_param(f) for f in kw_only_fields]
    func_builder.add_fn('__init__',
                        [self_name] + _init_params,
                        body_lines,
                        locals=locals,
                        return_type=None)


def _frozen_set_del_attr(cls, fields, func_builder):
    locals = {'__class__': cls,
              'FrozenInstanceError': FrozenInstanceError}
    condition = 'type(self) is __class__'
    if fields:
        condition += ' or name in {' + ', '.join(repr(f.name) for f in fields) + '}'

    func_builder.add_fn('__setattr__',
                        ('self', 'name', 'value'),
                        (f'  if {condition}:',
                          '   raise FrozenInstanceError(f"cannot assign to field {name!r}")',
                         f'  super(__class__, self).__setattr__(name, value)'),
                        locals=locals,
                        overwrite_error=True)
    func_builder.add_fn('__delattr__',
                        ('self', 'name'),
                        (f'  if {condition}:',
                          '   raise FrozenInstanceError(f"cannot delete field {name!r}")',
                         f'  super(__class__, self).__delattr__(name)'),
                        locals=locals,
                        overwrite_error=True)


def _is_classvar(a_type, typing):
    # This test uses a typing internal class, but it's the best way to
    # test if this is a ClassVar.
    return (a_type is typing.ClassVar
            or (type(a_type) is typing._GenericAlias
                and a_type.__origin__ is typing.ClassVar))


def _is_initvar(a_type, dataclasses):
    # The module we're checking against is the module we're
    # currently in (dataclasses.py).
    return (a_type is dataclasses.InitVar
            or type(a_type) is dataclasses.InitVar)

def _is_kw_only(a_type, dataclasses):
    return a_type is dataclasses.KW_ONLY


def _is_type(annotation, cls, a_module, a_type, is_type_predicate):
    # Given a type annotation string, does it refer to a_type in
    # a_module?  For example, when checking that annotation denotes a
    # ClassVar, then a_module is typing, and a_type is
    # typing.ClassVar.

    # It's possible to look up a_module given a_type, but it involves
    # looking in sys.modules (again!), and seems like a waste since
    # the caller already knows a_module.

    # - annotation is a string type annotation
    # - cls is the class that this annotation was found in
    # - a_module is the module we want to match
    # - a_type is the type in that module we want to match
    # - is_type_predicate is a function called with (obj, a_module)
    #   that determines if obj is of the desired type.

    # Since this test does not do a local namespace lookup (and
    # instead only a module (global) lookup), there are some things it
    # gets wrong.

    # With string annotations, cv0 will be detected as a ClassVar:
    #   CV = ClassVar
    #   @dataclass
    #   class C0:
    #     cv0: CV

    # But in this example cv1 will not be detected as a ClassVar:
    #   @dataclass
    #   class C1:
    #     CV = ClassVar
    #     cv1: CV

    # In C1, the code in this function (_is_type) will look up "CV" in
    # the module and not find it, so it will not consider cv1 as a
    # ClassVar.  This is a fairly obscure corner case, and the best
    # way to fix it would be to eval() the string "CV" with the
    # correct global and local namespaces.  However that would involve
    # a eval() penalty for every single field of every dataclass
    # that's defined.  It was judged not worth it.

    match = _MODULE_IDENTIFIER_RE.match(annotation)
    if match:
        ns = None
        module_name = match.group(1)
        if not module_name:
            # No module name, assume the class's module did
            # "from dataclasses import InitVar".
            ns = sys.modules.get(cls.__module__).__dict__
        else:
            # Look up module_name in the class's module.
            module = sys.modules.get(cls.__module__)
            if module and module.__dict__.get(module_name) is a_module:
                ns = sys.modules.get(a_type.__module__).__dict__
        if ns and is_type_predicate(ns.get(match.group(2)), a_module):
            return True
    return False


def _get_field(cls, a_name, a_type, default_kw_only):
    # Return a Field object for this field name and type.  ClassVars and
    # InitVars are also returned, but marked as such (see f._field_type).
    # default_kw_only is the value of kw_only to use if there isn't a field()
    # that defines it.

    # If the default value isn't derived from Field, then it's only a
    # normal default value.  Convert it to a Field().
    default = getattr(cls, a_name, MISSING)
    if isinstance(default, Field):
        f = default
    else:
        if isinstance(default, types.MemberDescriptorType):
            # This is a field in __slots__, so it has no default value.
            default = MISSING
        f = field(default=default)

    # Only at this point do we know the name and the type.  Set them.
    f.name = a_name
    f.type = a_type

    # Assume it's a normal field until proven otherwise.  We're next
    # going to decide if it's a ClassVar or InitVar, everything else
    # is just a normal field.
    f._field_type = _FIELD

    # In addition to checking for actual types here, also check for
    # string annotations.  get_type_hints() won't always work for us
    # (see https://github.com/python/typing/issues/508 for example),
    # plus it's expensive and would require an eval for every string
    # annotation.  So, make a best effort to see if this is a ClassVar
    # or InitVar using regex's and checking that the thing referenced
    # is actually of the correct type.

    # For the complete discussion, see https://bugs.python.org/issue33453

    # If typing has not been imported, then it's impossible for any
    # annotation to be a ClassVar.  So, only look for ClassVar if
    # typing has been imported by any module (not necessarily cls's
    # module).
    typing = sys.modules.get('typing')
    if typing:
        if (_is_classvar(a_type, typing)
            or (isinstance(f.type, str)
                and _is_type(f.type, cls, typing, typing.ClassVar,
                             _is_classvar))):
            f._field_type = _FIELD_CLASSVAR

    # If the type is InitVar, or if it's a matching string annotation,
    # then it's an InitVar.
    if f._field_type is _FIELD:
        # The module we're checking against is the module we're
        # currently in (dataclasses.py).
        dataclasses = sys.modules[__name__]
        if (_is_initvar(a_type, dataclasses)
            or (isinstance(f.type, str)
                and _is_type(f.type, cls, dataclasses, dataclasses.InitVar,
                             _is_initvar))):
            f._field_type = _FIELD_INITVAR

    # Validations for individual fields.  This is delayed until now,
    # instead of in the Field() constructor, since only here do we
    # know the field name, which allows for better error reporting.

    # Special restrictions for ClassVar and InitVar.
    if f._field_type in (_FIELD_CLASSVAR, _FIELD_INITVAR):
        if f.default_factory is not MISSING:
            raise TypeError(f'field {f.name} cannot have a '
                            'default factory')
        # Should I check for other field settings? default_factory
        # seems the most serious to check for.  Maybe add others.  For
        # example, how about init=False (or really,
        # init=<not-the-default-init-value>)?  It makes no sense for
        # ClassVar and InitVar to specify init=<anything>.

    # kw_only validation and assignment.
    if f._field_type in (_FIELD, _FIELD_INITVAR):
        # For real and InitVar fields, if kw_only wasn't specified use the
        # default value.
        if f.kw_only is MISSING:
            f.kw_only = default_kw_only
    else:
        # Make sure kw_only isn't set for ClassVars
        assert f._field_type is _FIELD_CLASSVAR
        if f.kw_only is not MISSING:
            raise TypeError(f'field {f.name} is a ClassVar but specifies '
                            'kw_only')

    # For real fields, disallow mutable defaults.  Use unhashable as a proxy
    # indicator for mutability.  Read the __hash__ attribute from the class,
    # not the instance.
    if f._field_type is _FIELD and f.default.__class__.__hash__ is None:
        raise ValueError(f'mutable default {type(f.default)} for field '
                         f'{f.name} is not allowed: use default_factory')

    return f

def _set_new_attribute(cls, name, value):
    # Never overwrites an existing attribute.  Returns True if the
    # attribute already exists.
    if name in cls.__dict__:
        return True
    setattr(cls, name, value)
    return False


# Decide if/how we're going to create a hash function.  Key is
# (unsafe_hash, eq, frozen, does-hash-exist).  Value is the action to
# take.  The common case is to do nothing, so instead of providing a
# function that is a no-op, use None to signify that.

def _hash_set_none(cls, fields, func_builder):
    # It's sort of a hack that I'm setting this here, instead of at
    # func_builder.add_fns_to_class time, but since this is an exceptional case
    # (it's not setting an attribute to a function, but to a scalar value),
    # just do it directly here.  I might come to regret this.
    cls.__hash__ = None

def _hash_add(cls, fields, func_builder):
    flds = [f for f in fields if (f.compare if f.hash is None else f.hash)]
    self_tuple = _tuple_str('self', flds)
    func_builder.add_fn('__hash__',
                        ('self',),
                        [f'  return hash({self_tuple})'],
                        unconditional_add=True)

def _hash_exception(cls, fields, func_builder):
    # Raise an exception.
    raise TypeError(f'Cannot overwrite attribute __hash__ '
                    f'in class {cls.__name__}')

#
#                +-------------------------------------- unsafe_hash?
#                |      +------------------------------- eq?
#                |      |      +------------------------ frozen?
#                |      |      |      +----------------  has-explicit-hash?
#                |      |      |      |
#                |      |      |      |        +-------  action
#                |      |      |      |        |
#                v      v      v      v        v
_hash_action = {(False, False, False, False): None,
                (False, False, False, True ): None,
                (False, False, True,  False): None,
                (False, False, True,  True ): None,
                (False, True,  False, False): _hash_set_none,
                (False, True,  False, True ): None,
                (False, True,  True,  False): _hash_add,
                (False, True,  True,  True ): None,
                (True,  False, False, False): _hash_add,
                (True,  False, False, True ): _hash_exception,
                (True,  False, True,  False): _hash_add,
                (True,  False, True,  True ): _hash_exception,
                (True,  True,  False, False): _hash_add,
                (True,  True,  False, True ): _hash_exception,
                (True,  True,  True,  False): _hash_add,
                (True,  True,  True,  True ): _hash_exception,
                }
# See https://bugs.python.org/issue32929#msg312829 for an if-statement
# version of this table.


def _process_class(cls, init, repr, eq, order, unsafe_hash, frozen,
                   match_args, kw_only, slots, weakref_slot):
    # Now that dicts retain insertion order, there's no reason to use
    # an ordered dict.  I am leveraging that ordering here, because
    # derived class fields overwrite base class fields, but the order
    # is defined by the base class, which is found first.
    fields = {}

    if cls.__module__ in sys.modules:
        globals = sys.modules[cls.__module__].__dict__
    else:
        # Theoretically this can happen if someone writes
        # a custom string to cls.__module__.  In which case
        # such dataclass won't be fully introspectable
        # (w.r.t. typing.get_type_hints) but will still function
        # correctly.
        globals = {}

    setattr(cls, _PARAMS, _DataclassParams(init, repr, eq, order,
                                           unsafe_hash, frozen,
                                           match_args, kw_only,
                                           slots, weakref_slot))

    # Find our base classes in reverse MRO order, and exclude
    # ourselves.  In reversed order so that more derived classes
    # override earlier field definitions in base classes.  As long as
    # we're iterating over them, see if all or any of them are frozen.
    any_frozen_base = False
    # By default `all_frozen_bases` is `None` to represent a case,
    # where some dataclasses does not have any bases with `_FIELDS`
    all_frozen_bases = None
    has_dataclass_bases = False
    for b in cls.__mro__[-1:0:-1]:
        # Only process classes that have been processed by our
        # decorator.  That is, they have a _FIELDS attribute.
        base_fields = getattr(b, _FIELDS, None)
        if base_fields is not None:
            has_dataclass_bases = True
            for f in base_fields.values():
                fields[f.name] = f
            if all_frozen_bases is None:
                all_frozen_bases = True
            current_frozen = getattr(b, _PARAMS).frozen
            all_frozen_bases = all_frozen_bases and current_frozen
            any_frozen_base = any_frozen_base or current_frozen

    # Annotations defined specifically in this class (not in base classes).
    #
    # Fields are found from cls_annotations, which is guaranteed to be
    # ordered.  Default values are from class attributes, if a field
    # has a default.  If the default value is a Field(), then it
    # contains additional info beyond (and possibly including) the
    # actual default value.  Pseudo-fields ClassVars and InitVars are
    # included, despite the fact that they're not real fields.  That's
    # dealt with later.
    cls_annotations = inspect.get_annotations(cls)

    # Now find fields in our class.  While doing so, validate some
    # things, and set the default values (as class attributes) where
    # we can.
    cls_fields = []
    # Get a reference to this module for the _is_kw_only() test.
    KW_ONLY_seen = False
    dataclasses = sys.modules[__name__]
    for name, type in cls_annotations.items():
        # See if this is a marker to change the value of kw_only.
        if (_is_kw_only(type, dataclasses)
            or (isinstance(type, str)
                and _is_type(type, cls, dataclasses, dataclasses.KW_ONLY,
                             _is_kw_only))):
            # Switch the default to kw_only=True, and ignore this
            # annotation: it's not a real field.
            if KW_ONLY_seen:
                raise TypeError(f'{name!r} is KW_ONLY, but KW_ONLY '
                                'has already been specified')
            KW_ONLY_seen = True
            kw_only = True
        else:
            # Otherwise it's a field of some type.
            cls_fields.append(_get_field(cls, name, type, kw_only))

    for f in cls_fields:
        fields[f.name] = f

        # If the class attribute (which is the default value for this
        # field) exists and is of type 'Field', replace it with the
        # real default.  This is so that normal class introspection
        # sees a real default value, not a Field.
        if isinstance(getattr(cls, f.name, None), Field):
            if f.default is MISSING:
                # If there's no default, delete the class attribute.
                # This happens if we specify field(repr=False), for
                # example (that is, we specified a field object, but
                # no default value).  Also if we're using a default
                # factory.  The class attribute should not be set at
                # all in the post-processed class.
                delattr(cls, f.name)
            else:
                setattr(cls, f.name, f.default)

    # Do we have any Field members that don't also have annotations?
    for name, value in cls.__dict__.items():
        if isinstance(value, Field) and not name in cls_annotations:
            raise TypeError(f'{name!r} is a field but has no type annotation')

    # Check rules that apply if we are derived from any dataclasses.
    if has_dataclass_bases:
        # Raise an exception if any of our bases are frozen, but we're not.
        if any_frozen_base and not frozen:
            raise TypeError('cannot inherit non-frozen dataclass from a '
                            'frozen one')

        # Raise an exception if we're frozen, but none of our bases are.
        if all_frozen_bases is False and frozen:
            raise TypeError('cannot inherit frozen dataclass from a '
                            'non-frozen one')

    # Remember all of the fields on our class (including bases).  This
    # also marks this class as being a dataclass.
    setattr(cls, _FIELDS, fields)

    # Was this class defined with an explicit __hash__?  Note that if
    # __eq__ is defined in this class, then python will automatically
    # set __hash__ to None.  This is a heuristic, as it's possible
    # that such a __hash__ == None was not auto-generated, but it's
    # close enough.
    class_hash = cls.__dict__.get('__hash__', MISSING)
    has_explicit_hash = not (class_hash is MISSING or
                             (class_hash is None and '__eq__' in cls.__dict__))

    # If we're generating ordering methods, we must be generating the
    # eq methods.
    if order and not eq:
        raise ValueError('eq must be true if order is true')

    # Include InitVars and regular fields (so, not ClassVars).  This is
    # initialized here, outside of the "if init:" test, because std_init_fields
    # is used with match_args, below.
    all_init_fields = [f for f in fields.values()
                       if f._field_type in (_FIELD, _FIELD_INITVAR)]
    (std_init_fields,
     kw_only_init_fields) = _fields_in_init_order(all_init_fields)

    func_builder = _FuncBuilder(globals)

    if init:
        # Does this class have a post-init function?
        has_post_init = hasattr(cls, _POST_INIT_NAME)

        _init_fn(all_init_fields,
                 std_init_fields,
                 kw_only_init_fields,
                 frozen,
                 has_post_init,
                 # The name to use for the "self"
                 # param in __init__.  Use "self"
                 # if possible.
                 '__dataclass_self__' if 'self' in fields
                 else 'self',
                 func_builder,
                 slots,
                 )

    _set_new_attribute(cls, '__replace__', _replace)

    # Get the fields as a list, and include only real fields.  This is
    # used in all of the following methods.
    field_list = [f for f in fields.values() if f._field_type is _FIELD]

    if repr:
        flds = [f for f in field_list if f.repr]
        func_builder.add_fn('__repr__',
                            ('self',),
                            ['  return f"{self.__class__.__qualname__}(' +
                             ', '.join([f"{f.name}={{self.{f.name}!r}}"
                                        for f in flds]) + ')"'],
                            locals={'__dataclasses_recursive_repr': recursive_repr},
                            decorator="@__dataclasses_recursive_repr()")

    if eq:
        # Create __eq__ method.  There's no need for a __ne__ method,
        # since python will call __eq__ and negate it.
        cmp_fields = (field for field in field_list if field.compare)
        terms = [f'self.{field.name}==other.{field.name}' for field in cmp_fields]
        field_comparisons = ' and '.join(terms) or 'True'
        func_builder.add_fn('__eq__',
                            ('self', 'other'),
                            [ '  if self is other:',
                              '   return True',
                              '  if other.__class__ is self.__class__:',
                             f'   return {field_comparisons}',
                              '  return NotImplemented'])

    if order:
        # Create and set the ordering methods.
        flds = [f for f in field_list if f.compare]
        self_tuple = _tuple_str('self', flds)
        other_tuple = _tuple_str('other', flds)
        for name, op in [('__lt__', '<'),
                         ('__le__', '<='),
                         ('__gt__', '>'),
                         ('__ge__', '>='),
                         ]:
            # Create a comparison function.  If the fields in the object are
            # named 'x' and 'y', then self_tuple is the string
            # '(self.x,self.y)' and other_tuple is the string
            # '(other.x,other.y)'.
            func_builder.add_fn(name,
                            ('self', 'other'),
                            [ '  if other.__class__ is self.__class__:',
                             f'   return {self_tuple}{op}{other_tuple}',
                              '  return NotImplemented'],
                            overwrite_error='Consider using functools.total_ordering')

    if frozen:
        _frozen_set_del_attr(cls, field_list, func_builder)

    # Decide if/how we're going to create a hash function.
    hash_action = _hash_action[bool(unsafe_hash),
                               bool(eq),
                               bool(frozen),
                               has_explicit_hash]
    if hash_action:
        cls.__hash__ = hash_action(cls, field_list, func_builder)

    # Generate the methods and add them to the class.  This needs to be done
    # before the __doc__ logic below, since inspect will look at the __init__
    # signature.
    func_builder.add_fns_to_class(cls)

    if not getattr(cls, '__doc__'):
        # Create a class doc-string.
        try:
            # In some cases fetching a signature is not possible.
            # But, we surely should not fail in this case.
            text_sig = str(inspect.signature(cls)).replace(' -> None', '')
        except (TypeError, ValueError):
            text_sig = ''
        cls.__doc__ = (cls.__name__ + text_sig)

    if match_args:
        # I could probably compute this once.
        _set_new_attribute(cls, '__match_args__',
                           tuple(f.name for f in std_init_fields))

    # It's an error to specify weakref_slot if slots is False.
    if weakref_slot and not slots:
        raise TypeError('weakref_slot is True but slots is False')
    if slots:
        cls = _add_slots(cls, frozen, weakref_slot)

    abc.update_abstractmethods(cls)

    return cls


# _dataclass_getstate and _dataclass_setstate are needed for pickling frozen
# classes with slots.  These could be slightly more performant if we generated
# the code instead of iterating over fields.  But that can be a project for
# another day, if performance becomes an issue.
def _dataclass_getstate(self):
    return [getattr(self, f.name) for f in fields(self)]


def _dataclass_setstate(self, state):
    for field, value in zip(fields(self), state):
        # use setattr because dataclass may be frozen
        object.__setattr__(self, field.name, value)


def _get_slots(cls):
    match cls.__dict__.get('__slots__'):
        # `__dictoffset__` and `__weakrefoffset__` can tell us whether
        # the base type has dict/weakref slots, in a way that works correctly
        # for both Python classes and C extension types. Extension types
        # don't use `__slots__` for slot creation
        case None:
            slots = []
            if getattr(cls, '__weakrefoffset__', -1) != 0:
                slots.append('__weakref__')
            if getattr(cls, '__dictoffset__', -1) != 0:
                slots.append('__dict__')
            yield from slots
        case str(slot):
            yield slot
        # Slots may be any iterable, but we cannot handle an iterator
        # because it will already be (partially) consumed.
        case iterable if not hasattr(iterable, '__next__'):
            yield from iterable
        case _:
            raise TypeError(f"Slots of '{cls.__name__}' cannot be determined")


def _update_func_cell_for__class__(f, oldcls, newcls):
    # Returns True if we update a cell, else False.
    if f is None:
        # f will be None in the case of a property where not all of
        # fget, fset, and fdel are used.  Nothing to do in that case.
        return False
    try:
        idx = f.__code__.co_freevars.index("__class__")
    except ValueError:
        # This function doesn't reference __class__, so nothing to do.
        return False
    # Fix the cell to point to the new class, if it's already pointing
    # at the old class.
    closure = f.__closure__[idx]

    try:
        contents = closure.cell_contents
    except ValueError:
        # Cell is empty
        return False

    # This check makes it so we avoid updating an incorrect cell if the
    # class body contains a function that was defined in a different class.
    if contents is oldcls:
        closure.cell_contents = newcls
        return True
    return False


def _add_slots(cls, is_frozen, weakref_slot):
    # Need to create a new class, since we can't set __slots__
    #  after a class has been created.

    # Make sure __slots__ isn't already set.
    if '__slots__' in cls.__dict__:
        raise TypeError(f'{cls.__name__} already specifies __slots__')

    # Create a new dict for our new class.
    cls_dict = dict(cls.__dict__)
    field_names = tuple(f.name for f in fields(cls))
    # Make sure slots don't overlap with those in base classes.
    inherited_slots = set(
        itertools.chain.from_iterable(map(_get_slots, cls.__mro__[1:-1]))
    )
    # The slots for our class.  Remove slots from our base classes.  Add
    # '__weakref__' if weakref_slot was given, unless it is already present.
    cls_dict["__slots__"] = tuple(
        itertools.filterfalse(
            inherited_slots.__contains__,
            itertools.chain(
                # gh-93521: '__weakref__' also needs to be filtered out if
                # already present in inherited_slots
                field_names, ('__weakref__',) if weakref_slot else ()
            )
        ),
    )

    for field_name in field_names:
        # Remove our attributes, if present. They'll still be
        #  available in _MARKER.
        cls_dict.pop(field_name, None)

    # Remove __dict__ itself.
    cls_dict.pop('__dict__', None)

    # Clear existing `__weakref__` descriptor, it belongs to a previous type:
    cls_dict.pop('__weakref__', None)  # gh-102069

    # And finally create the class.
    qualname = getattr(cls, '__qualname__', None)
    newcls = type(cls)(cls.__name__, cls.__bases__, cls_dict)
    if qualname is not None:
        newcls.__qualname__ = qualname

    if is_frozen:
        # Need this for pickling frozen classes with slots.
        if '__getstate__' not in cls_dict:
            newcls.__getstate__ = _dataclass_getstate
        if '__setstate__' not in cls_dict:
            newcls.__setstate__ = _dataclass_setstate

    # Fix up any closures which reference __class__.  This is used to
    # fix zero argument super so that it points to the correct class
    # (the newly created one, which we're returning) and not the
    # original class.  We can break out of this loop as soon as we
    # make an update, since all closures for a class will share a
    # given cell.
    for member in newcls.__dict__.values():
        # If this is a wrapped function, unwrap it.
        member = inspect.unwrap(member)

        if isinstance(member, types.FunctionType):
            if _update_func_cell_for__class__(member, cls, newcls):
                break
        elif isinstance(member, property):
            if (_update_func_cell_for__class__(member.fget, cls, newcls)
                or _update_func_cell_for__class__(member.fset, cls, newcls)
                or _update_func_cell_for__class__(member.fdel, cls, newcls)):
                break

    return newcls


def dataclass(cls=None, /, *, init=True, repr=True, eq=True, order=False,
              unsafe_hash=False, frozen=False, match_args=True,
              kw_only=False, slots=False, weakref_slot=False):
    """Add dunder methods based on the fields defined in the class.

    Examines PEP 526 __annotations__ to determine fields.

    If init is true, an __init__() method is added to the class. If repr
    is true, a __repr__() method is added. If order is true, rich
    comparison dunder methods are added. If unsafe_hash is true, a
    __hash__() method is added. If frozen is true, fields may not be
    assigned to after instance creation. If match_args is true, the
    __match_args__ tuple is added. If kw_only is true, then by default
    all fields are keyword-only. If slots is true, a new class with a
    __slots__ attribute is returned.
    """

    def wrap(cls):
        return _process_class(cls, init, repr, eq, order, unsafe_hash,
                              frozen, match_args, kw_only, slots,
                              weakref_slot)

    # See if we're being called as @dataclass or @dataclass().
    if cls is None:
        # We're called with parens.
        return wrap

    # We're called as @dataclass without parens.
    return wrap(cls)


def fields(class_or_instance):
    """Return a tuple describing the fields of this dataclass.

    Accepts a dataclass or an instance of one. Tuple elements are of
    type Field.
    """

    # Might it be worth caching this, per class?
    try:
        fields = getattr(class_or_instance, _FIELDS)
    except AttributeError:
        raise TypeError('must be called with a dataclass type or instance') from None

    # Exclude pseudo-fields.  Note that fields is sorted by insertion
    # order, so the order of the tuple is as the fields were defined.
    return tuple(f for f in fields.values() if f._field_type is _FIELD)


def _is_dataclass_instance(obj):
    """Returns True if obj is an instance of a dataclass."""
    return hasattr(type(obj), _FIELDS)


def is_dataclass(obj):
    """Returns True if obj is a dataclass or an instance of a
    dataclass."""
    cls = obj if isinstance(obj, type) else type(obj)
    return hasattr(cls, _FIELDS)


def asdict(obj, *, dict_factory=dict):
    """Return the fields of a dataclass instance as a new dictionary mapping
    field names to field values.

    Example usage::

      @dataclass
      class C:
          x: int
          y: int

      c = C(1, 2)
      assert asdict(c) == {'x': 1, 'y': 2}

    If given, 'dict_factory' will be used instead of built-in dict.
    The function applies recursively to field values that are
    dataclass instances. This will also look into built-in containers:
    tuples, lists, and dicts. Other objects are copied with 'copy.deepcopy()'.
    """
    if not _is_dataclass_instance(obj):
        raise TypeError("asdict() should be called on dataclass instances")
    return _asdict_inner(obj, dict_factory)


def _asdict_inner(obj, dict_factory):
    obj_type = type(obj)
    if obj_type in _ATOMIC_TYPES:
        return obj
    elif hasattr(obj_type, _FIELDS):
        # dataclass instance: fast path for the common case
        if dict_factory is dict:
            return {
                f.name: _asdict_inner(getattr(obj, f.name), dict)
                for f in fields(obj)
            }
        else:
            return dict_factory([
                (f.name, _asdict_inner(getattr(obj, f.name), dict_factory))
                for f in fields(obj)
            ])
    # handle the builtin types first for speed; subclasses handled below
    elif obj_type is list:
        return [_asdict_inner(v, dict_factory) for v in obj]
    elif obj_type is dict:
        return {
            _asdict_inner(k, dict_factory): _asdict_inner(v, dict_factory)
            for k, v in obj.items()
        }
    elif obj_type is tuple:
        return tuple([_asdict_inner(v, dict_factory) for v in obj])
    elif issubclass(obj_type, tuple):
        if hasattr(obj, '_fields'):
            # obj is a namedtuple.  Recurse into it, but the returned
            # object is another namedtuple of the same type.  This is
            # similar to how other list- or tuple-derived classes are
            # treated (see below), but we just need to create them
            # differently because a namedtuple's __init__ needs to be
            # called differently (see bpo-34363).

            # I'm not using namedtuple's _asdict()
            # method, because:
            # - it does not recurse in to the namedtuple fields and
            #   convert them to dicts (using dict_factory).
            # - I don't actually want to return a dict here.  The main
            #   use case here is json.dumps, and it handles converting
            #   namedtuples to lists.  Admittedly we're losing some
            #   information here when we produce a json list instead of a
            #   dict.  Note that if we returned dicts here instead of
            #   namedtuples, we could no longer call asdict() on a data
            #   structure where a namedtuple was used as a dict key.
            return obj_type(*[_asdict_inner(v, dict_factory) for v in obj])
        else:
            return obj_type(_asdict_inner(v, dict_factory) for v in obj)
    elif issubclass(obj_type, dict):
        if hasattr(obj_type, 'default_factory'):
            # obj is a defaultdict, which has a different constructor from
            # dict as it requires the default_factory as its first arg.
            result = obj_type(obj.default_factory)
            for k, v in obj.items():
                result[_asdict_inner(k, dict_factory)] = _asdict_inner(v, dict_factory)
            return result
        return obj_type((_asdict_inner(k, dict_factory),
                         _asdict_inner(v, dict_factory))
                        for k, v in obj.items())
    elif issubclass(obj_type, list):
        # Assume we can create an object of this type by passing in a
        # generator
        return obj_type(_asdict_inner(v, dict_factory) for v in obj)
    else:
        return copy.deepcopy(obj)


def astuple(obj, *, tuple_factory=tuple):
    """Return the fields of a dataclass instance as a new tuple of field values.

    Example usage::

      @dataclass
      class C:
          x: int
          y: int

      c = C(1, 2)
      assert astuple(c) == (1, 2)

    If given, 'tuple_factory' will be used instead of built-in tuple.
    The function applies recursively to field values that are
    dataclass instances. This will also look into built-in containers:
    tuples, lists, and dicts. Other objects are copied with 'copy.deepcopy()'.
    """

    if not _is_dataclass_instance(obj):
        raise TypeError("astuple() should be called on dataclass instances")
    return _astuple_inner(obj, tuple_factory)


def _astuple_inner(obj, tuple_factory):
    if type(obj) in _ATOMIC_TYPES:
        return obj
    elif _is_dataclass_instance(obj):
        return tuple_factory([
            _astuple_inner(getattr(obj, f.name), tuple_factory)
            for f in fields(obj)
        ])
    elif isinstance(obj, tuple) and hasattr(obj, '_fields'):
        # obj is a namedtuple.  Recurse into it, but the returned
        # object is another namedtuple of the same type.  This is
        # similar to how other list- or tuple-derived classes are
        # treated (see below), but we just need to create them
        # differently because a namedtuple's __init__ needs to be
        # called differently (see bpo-34363).
        return type(obj)(*[_astuple_inner(v, tuple_factory) for v in obj])
    elif isinstance(obj, (list, tuple)):
        # Assume we can create an object of this type by passing in a
        # generator (which is not true for namedtuples, handled
        # above).
        return type(obj)(_astuple_inner(v, tuple_factory) for v in obj)
    elif isinstance(obj, dict):
        obj_type = type(obj)
        if hasattr(obj_type, 'default_factory'):
            # obj is a defaultdict, which has a different constructor from
            # dict as it requires the default_factory as its first arg.
            result = obj_type(getattr(obj, 'default_factory'))
            for k, v in obj.items():
                result[_astuple_inner(k, tuple_factory)] = _astuple_inner(v, tuple_factory)
            return result
        return obj_type((_astuple_inner(k, tuple_factory), _astuple_inner(v, tuple_factory))
                          for k, v in obj.items())
    else:
        return copy.deepcopy(obj)


def make_dataclass(cls_name, fields, *, bases=(), namespace=None, init=True,
                   repr=True, eq=True, order=False, unsafe_hash=False,
                   frozen=False, match_args=True, kw_only=False, slots=False,
                   weakref_slot=False, module=None):
    """Return a new dynamically created dataclass.

    The dataclass name will be 'cls_name'.  'fields' is an iterable
    of either (name), (name, type) or (name, type, Field) objects. If type is
    omitted, use the string 'typing.Any'.  Field objects are created by
    the equivalent of calling 'field(name, type [, Field-info])'.::

      C = make_dataclass('C', ['x', ('y', int), ('z', int, field(init=False))], bases=(Base,))

    is equivalent to::

      @dataclass
      class C(Base):
          x: 'typing.Any'
          y: int
          z: int = field(init=False)

    For the bases and namespace parameters, see the builtin type() function.

    The parameters init, repr, eq, order, unsafe_hash, frozen, match_args, kw_only,
    slots, and weakref_slot are passed to dataclass().

    If module parameter is defined, the '__module__' attribute of the dataclass is
    set to that value.
    """

    if namespace is None:
        namespace = {}

    # While we're looking through the field names, validate that they
    # are identifiers, are not keywords, and not duplicates.
    seen = set()
    annotations = {}
    defaults = {}
    for item in fields:
        if isinstance(item, str):
            name = item
            tp = 'typing.Any'
        elif len(item) == 2:
            name, tp, = item
        elif len(item) == 3:
            name, tp, spec = item
            defaults[name] = spec
        else:
            raise TypeError(f'Invalid field: {item!r}')

        if not isinstance(name, str) or not name.isidentifier():
            raise TypeError(f'Field names must be valid identifiers: {name!r}')
        if keyword.iskeyword(name):
            raise TypeError(f'Field names must not be keywords: {name!r}')
        if name in seen:
            raise TypeError(f'Field name duplicated: {name!r}')

        seen.add(name)
        annotations[name] = tp

    # Update 'ns' with the user-supplied namespace plus our calculated values.
    def exec_body_callback(ns):
        ns.update(namespace)
        ns.update(defaults)
        ns['__annotations__'] = annotations

    # We use `types.new_class()` instead of simply `type()` to allow dynamic creation
    # of generic dataclasses.
    cls = types.new_class(cls_name, bases, {}, exec_body_callback)

    # For pickling to work, the __module__ variable needs to be set to the frame
    # where the dataclass is created.
    if module is None:
        try:
            module = sys._getframemodulename(1) or '__main__'
        except AttributeError:
            try:
                module = sys._getframe(1).f_globals.get('__name__', '__main__')
            except (AttributeError, ValueError):
                pass
    if module is not None:
        cls.__module__ = module

    # Apply the normal decorator.
    return dataclass(cls, init=init, repr=repr, eq=eq, order=order,
                     unsafe_hash=unsafe_hash, frozen=frozen,
                     match_args=match_args, kw_only=kw_only, slots=slots,
                     weakref_slot=weakref_slot)


def replace(obj, /, **changes):
    """Return a new object replacing specified fields with new values.

    This is especially useful for frozen classes.  Example usage::

      @dataclass(frozen=True)
      class C:
          x: int
          y: int

      c = C(1, 2)
      c1 = replace(c, x=3)
      assert c1.x == 3 and c1.y == 2
    """
    if not _is_dataclass_instance(obj):
        raise TypeError("replace() should be called on dataclass instances")
    return _replace(obj, **changes)


def _replace(self, /, **changes):
    # We're going to mutate 'changes', but that's okay because it's a
    # new dict, even if called with 'replace(self, **my_changes)'.

    # It's an error to have init=False fields in 'changes'.
    # If a field is not in 'changes', read its value from the provided 'self'.

    for f in getattr(self, _FIELDS).values():
        # Only consider normal fields or InitVars.
        if f._field_type is _FIELD_CLASSVAR:
            continue

        if not f.init:
            # Error if this field is specified in changes.
            if f.name in changes:
                raise TypeError(f'field {f.name} is declared with '
                                f'init=False, it cannot be specified with '
                                f'replace()')
            continue

        if f.name not in changes:
            if f._field_type is _FIELD_INITVAR and f.default is MISSING:
                raise TypeError(f"InitVar {f.name!r} "
                                f'must be specified with replace()')
            changes[f.name] = getattr(self, f.name)

    # Create the new object, which calls __init__() and
    # __post_init__() (if defined), using all of the init fields we've
    # added and/or left in 'changes'.  If there are values supplied in
    # changes that aren't fields, this will correctly raise a
    # TypeError.
    return self.__class__(**changes)
