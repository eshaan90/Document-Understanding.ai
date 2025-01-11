import os
import json
import itertools
import streamlit as st

def get_name_from_path(path):
    return os.path.basename(path).split(".")[0]


def save_to_file(table_predictions, result_path, output_file_type='json'):
    if output_file_type == "json":
        with open(result_path, "w+", encoding="utf-8") as f:
            json.dump(table_predictions, f, ensure_ascii=False)


def check_if_any_file_exists(directory):
    """Checks if any file exists in the given directory."""
    return any(os.path.isfile(os.path.join(directory, f)) for f in os.listdir(directory))



def paginator(label, list_items, items_per_page=5, on_sidebar=True):
    """Lets the user paginate a set of items.
    Parameters
    ----------
    label : str
        The label to display over the pagination widget.
    list_items : Iterator[Any]
        The items to display in the paginator.
    items_per_page: int
        The number of items to display per page.
    on_sidebar: bool
        Whether to display the paginator widget on the sidebar.
        
    Returns
    -------
    Iterator[Tuple[str, Any]]
        An iterator over *only the items on that page*, including
        the item's caption.
    Example
    -------
    This shows how to display a few pages of fruit.
    >>> fruit_list = [
    ...     'Kiwifruit', 'Honeydew', 'Cherry', 'Honeyberry', 'Pear',
    ...     'Apple', 'Nectarine', 'Soursop', 'Pineapple', 'Satsuma',
    ...     'Fig', 'Huckleberry', 'Coconut', 'Plantain', 'Jujube',
    ...     'Guava', 'Clementine', 'Grape', 'Tayberry', 'Salak',
    ...     'Raspberry', 'Loquat', 'Nance', 'Peach', 'Akee'
    ... ]
    ...
    ... for i, fruit in paginator("Select a fruit page", fruit_list):
    ...     st.write('%s. **%s**' % (i, fruit))
    """

    # Figure out where to display the paginator
    if on_sidebar:
        location = st.sidebar.empty()
    else:
        location = st.empty()

    # Display a pagination selectbox in the specified location.
    # total_items = len(items)
    # n_pages = len(items)    

    n_pages = (len(list_items) - 1) // items_per_page + 1
    if isinstance(list_items, dict):
        list_items=list(list_items.items())
    page_format_func = lambda i: "Page %s" % i
    page_number = location.selectbox(label, range(n_pages), index=0, format_func=page_format_func)
    st.write(page_number,items_per_page)
    # Iterate over the items in the page to let the user display them.
    min_index = page_number * items_per_page
    max_index = min_index + items_per_page
    
    return itertools.islice(enumerate(list_items), min_index, max_index)
