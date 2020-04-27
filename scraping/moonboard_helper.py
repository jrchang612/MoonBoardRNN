# Authors: Aaron Wu, Howard Tai

# Script containing helper functions for accessing and navigating the MoonBoard web-site

import os
import copy
import json
import time
import string
import pickle

from bs4 import BeautifulSoup
from selenium import webdriver


# ----------------------------------------------------------------------------------------------------------------------
# Selenium processing functions
# ----------------------------------------------------------------------------------------------------------------------
def load_browser(driver_path):
    """
    Loads an incognito Chrome browser object, given a chromedriver path
    """
    option = webdriver.ChromeOptions()
    option.add_argument('--incognito')

    browser = webdriver.Chrome(executable_path=driver_path, options=option)
    browser.set_window_size(1500, 910)
    return browser


def find_element_attr(browser, tag_name, attribute):
    """
    Helper function for finding HTML element attributes
    """
    values = None
    try:
        elems = browser.find_elements_by_tag_name(tag_name)
        values = [e.get_attribute(attribute) for e in elems]
        if all(v is None for v in values):
            values = None
    except:
        pass
    if values is None:
        print('Failed to find ' + str(attribute))
    return values


def find_element_text(browser, tag_name):
    """
    Finds text attached to HTML element
    """
    values = None
    try:
        elems = browser.find_elements_by_tag_name(tag_name)
        values = [e.text for e in elems]
        if all(v is None for v in values):
            values = None
    except:
        pass
    if values is None:
        print('Failed to find ' + str(tag_name))
    return values


def find_element(browser, tag_name, attribute, value, num_tries=10, sleep_val=1):
    """
    Finds elements of a given tag name, attribute, and value on a browser page
    """
    elem = None
    for i in range(num_tries):
        try:
            elems = browser.find_elements_by_tag_name(tag_name)
            for e in elems:
                if e.get_attribute(attribute) == value:
                    elem = e
                    break
            if elem is not None:
                break
        except:
            time.sleep(sleep_val)
            continue
    if elem is None:
        print('Failed to find ' + str(attribute) + ' ' + str(value))
    return elem


def find_and_click(browser, tag_name, attribute, value, num_tries=10, sleep_val=1):
    """
    Finds a specific HTML element and clicks on it
    """
    elem = None
    for i in range(num_tries):
        try:
            elem = find_element(browser, tag_name, attribute, value)
            if elem is not None:
                elem.click()
                break
        except:
            time.sleep(sleep_val)
            continue
    if elem is None:
        print('Failed to click')
    return elem


def find_text(browser, tag_name, text, num_tries=10, sleep_val=1):
    """
    Finds text corresponding to specific tag name of HTML object
    """
    elem = None
    for i in range(num_tries):
        try:
            elems = browser.find_elements_by_tag_name(tag_name)
            for e in elems:
                if e.text == text:
                    elem = e
                    break
            if elem is not None:
                break
        except:
            time.sleep(sleep_val)
            continue
    if elem is None:
        print('Failed to find ' + str(text))
    return elem


def get_elem_set(browser, tag_name, attr_dict, num_tries=10, sleep_val=1):
    """
    Given a browser page, find all the HTML elements of a given tag name and attribute
    """
    elem_set = []
    for i in range(num_tries):
        try:
            elems = browser.find_elements_by_tag_name(tag_name)
            for e in elems:
                if all(e.get_attribute(attr) == attr_dict[attr] for attr in attr_dict):
                    elem_set.append(e)
            if len(elem_set) > 0:  # Non-empty element set
                break
        except:
            time.sleep(sleep_val)
            continue
    return elem_set


def check_fail_url(browser, url):
    """
    Checks if a specific problem link is broken
    """
    browser.get(url)
    elems = find_element(browser, 'span', 'class', 'field-validation-error', num_tries=1)
    if elems is not None:
        return True
    return False


# ----------------------------------------------------------------------------------------------------------------------
# General utility functions
# ----------------------------------------------------------------------------------------------------------------------
def save_pickle(data, file_name):
    """
    Saves data as pickle format
    """
    with open(file_name, 'wb') as f:
        pickle.dump(data, f)
    return None


def load_pickle(file_name):
    """
    Loads data from pickle format
    """
    with open(file_name, 'rb') as f:
        return pickle.load(f)


def remove_duplicates(repeats_data):
    """
    Removes repeats from a list
    """
    repeats = []
    for r in repeats_data:
        if r not in repeats:
            repeats.append(r)
    return repeats


# ----------------------------------------------------------------------------------------------------------------------
# Login to MoonBoard site
# ----------------------------------------------------------------------------------------------------------------------
def click_login_area(browser):
    """
    Clicks on login area to pull up username and password field
    """
    login_elem = None
    a_elems = browser.find_elements_by_tag_name('a')
    for a in a_elems:
        if a.text == 'LOGIN/REGISTER':
            login_elem = a
            break
    if login_elem is None:
        print('Failed to find Login Button')
    else:
        login_elem.click()
    return login_elem


def click_login_button(browser):
    """
    Clicks login button
    """
    return find_and_click(browser, 'button', 'type', 'submit')


def input_user_pass_login(browser, username, password):
    """
    Populates username and password fields after accessing login area
    """
    username_elem = None
    password_elem = None
    input_elems = browser.find_elements_by_tag_name('input')

    # Iterate through input elements
    for i in input_elems:
        if i.get_attribute('placeholder') == 'Username':
            username_elem = i
        if i.get_attribute('placeholder') == 'Password':
            password_elem = i

    # Check that valid elements are returned
    if username_elem is None:
        print('Failed to find username field')
    if password_elem is None:
        print('Failed to find password field')

    # Populate fields
    if username_elem is not None and password_elem is not None:
        username_elem.send_keys(username)
        password_elem.send_keys(password)

    return username_elem, password_elem


def loginMoonBoard(browser, url='', username='', password=''):
    """
    Logs in to MoonBoard site
    """
    browser.get(url)

    # Get login element
    login_elem = click_login_area(browser)
    if login_elem is None:
        return None

    # Fill in credentials and login
    username_elem, password_elem = input_user_pass_login(browser, username, password)
    login_button = click_login_button(browser)

    if username_elem is None or password_elem is None or login_button is None:
        return None

    return login_button


# ----------------------------------------------------------------------------------------------------------------------
# Navigate to problems view
# ----------------------------------------------------------------------------------------------------------------------
def click_view_problems(browser):
    """
    Accesses 'View' under 'Problems' sidebar
    """
    click_problems = find_and_click(browser, 'a', 'id', 'lProblems')
    click_view = find_and_click(browser, 'li', 'id', 'm-viewproblem')
    return None

def click_holdsetup(browser, holdsetup='MoonBoard 2016'):
    """
    Selects the proper hold configuration (default MoonBoard 2016)
    """
    target_elem = None

    # Finds hold setup dropdown
    elems = browser.find_elements_by_tag_name('select')
    for e in elems:
        if e.get_attribute('id') == 'Holdsetup':
            target_elem = e
            break
    if target_elem is None:
        print('Failed to find Holdsetup')
        return target_elem

    # Selects appropriate dropdown item
    elems = target_elem.find_elements_by_tag_name('option')
    target_elem = None
    for e in elems:
        if e.text == holdsetup:
            target_elem = e
    if target_elem is None:
        print('Failed to find ' + holdsetup)
        return target_elem

    # Select hold configuration
    target_elem.click()
    return target_elem


# ----------------------------------------------------------------------------------------------------------------------
# Click through pages
# ----------------------------------------------------------------------------------------------------------------------
def get_current_page(browser):
    """
    Gets the index of the current page of routes (bottom bar)
    """
    pager_elem = None
    page_elem = None

    # Pull elements with 'div' tag
    elems = browser.find_elements_by_tag_name('div')
    for e in elems:
        if e.get_attribute('data-role') == 'pager':
            pager_elem = e
            break
    if pager_elem is None:
        print('Failed to find pager')
        return pager_elem

    # Pull elements with 'span' tag
    page_elems = pager_elem.find_elements_by_tag_name('span')
    for e in page_elems:
        if e.get_attribute('class') == 'k-state-selected':
            page_elem = e
            break
    if page_elem is None:
        print('Failed to find page')
        return page_elem

    return int(page_elem.text)


def click_next_page(browser, current_page=1):
    """
    Clicks on button to access next page of routes
    """
    next_page = current_page + 1
    page_elem = find_and_click(browser, 'a', 'data-page', str(next_page))
    return page_elem


# ----------------------------------------------------------------------------------------------------------------------
# Acquire problem metadata
# ----------------------------------------------------------------------------------------------------------------------
def get_problems(browser):
    """
    For a single page, get problem IDs and problem objects
    """
    problems = []
    data_ids = []
    elems = browser.find_elements_by_tag_name('tr')

    for e in elems:
        uid = e.get_attribute('data-uid')
        check1 = uid is not None
        check2 = e.get_attribute('onclick') == 'problemSelected();'
        if check1 and check2:
            data_ids.append(uid)
            problems.append(e)
    
    if len(problems) == 0:
        raise ValueError('No problem found in this page!')

    return problems, data_ids


def get_problem_meta(problem):
    """
    Finds metadata tags from a problem object
    """
    meta = {}

    h3 = problem.find_elements_by_tag_name('h3')[0]
    meta['problem_name'] = h3.text
    meta['info'] = [p.text for p in problem.find_elements_by_tag_name('p')]
    meta['url'] = h3.find_elements_by_tag_name('a')[0].get_attribute('href')

    # Rating information
    stars = [star.get_attribute('src') for star in problem.find_elements_by_tag_name('img')]
    stars = [star for star in stars if 'star' in star]
    meta['num_empty'] = len([star for star in stars if 'empty' in star])
    meta['num_stars'] = len(stars) - meta['num_empty']

    return meta


# ----------------------------------------------------------------------------------------------------------------------
# Functions to process specific problems
# ----------------------------------------------------------------------------------------------------------------------
def click_next_repeats(browser, current_page):
    """
    Click on next page of repeats information in specific problem view
    """
    next_page = current_page + 1
    pager = None
    clicked = False

    # Click on page number (bottom bar)
    spans = browser.find_elements_by_tag_name('span')
    for s in spans:
        if s.get_attribute('class') == 'k-link k-pager-nav':
            pager = s
    if pager is None:
        return clicked
    pager.click()

    # Click on next page (expanded page numbers)
    items = browser.find_elements_by_tag_name('a')
    for i in items:
        if i.get_attribute('data-page') == str(next_page):
            i.click()
            return True

    return False


def get_repeats_data_for_problem(browser, num_tries=20, sleep_val=1):
    """
    Scrapes a specific problem's repeat info
    """
    repeats_data = []
    attr_dict = {'class': 'repeats'}
    current_page = 1

    while True:
        repeats_add = None  # Initializing repeat info object

        # For a given page, get repeats info
        for x in range(num_tries):
            try:
                repeats = get_elem_set(browser, 'div', attr_dict)  # Repeat entries on single page

                for repeat in repeats:
                    h3_item = repeat.find_elements_by_tag_name('h3')[0]
                    user_name = h3_item.text
                    uid = h3_item.find_elements_by_tag_name('a')[0].get_attribute('href').split('/Account/Profile/')[1]

                    repeats_add = [r.text for r in repeat.find_elements_by_tag_name('p')] + [user_name, uid]
                    repeats_data.append(repeats_add)

                # Stop trying if successful
                break
            except:
                time.sleep(sleep_val)
                continue

        # Try clicking on next page of repeats info
        clicked = False
        for x in range(num_tries):
            try:
                clicked = click_next_repeats(browser, current_page)
                break
            except:
                time.sleep(sleep_val)
                continue

        # If clicking on next page is unsuccessful
        if not clicked:
            break
        current_page += 1

    # Remove duplicate repeat info
    repeats_data = remove_duplicates(repeats_data)
    return repeats_data


def get_target_script(browser):
    """
    Returns javascript code containing hold patterns
    """
    html = browser.page_source
    soup = BeautifulSoup(html, 'html.parser')
    scripts = soup.find_all("script")
    result = ''
    for s in scripts:
        if len(s.contents) != 0:
            if 'JSON' in s.contents[0]:
                result = s.contents[0]
    return result


def parse_target_json(text, start="var problem = JSON.parse('", end="');"):
    """
    Formats input javascript string into JSON
    """
    json_dict = json.loads(text.split(start)[1].split(end)[0])
    return json_dict

def parse_moves_and_others(json_dict, tmp_problem):
    """
    Returns set of moves, grade, user grade, isBenchmark, repeat, 
    and setter.
    """
    tmp_problem['moves'] = json_dict['Moves']
    tmp_problem['grade'] = json_dict['Grade']
    tmp_problem['UserGrade'] = json_dict['UserGrade']
    tmp_problem['isBenchmark'] = json_dict['IsBenchmark']
    tmp_problem['repeats'] = json_dict['Repeats']
    tmp_problem['ProblemType'] = json_dict['ProblemType']
    tmp_problem['IsMaster'] = json_dict['IsMaster']
    tmp_problem['setter'] = json_dict['Setter']
    return tmp_problem

def parse_moves(json_dict):
    """
    Returns set of moves for a problem
    """
    return json_dict['Moves']

# ----------------------------------------------------------------------------------------------------------------------
# Batch processing functions
# ----------------------------------------------------------------------------------------------------------------------
def get_num_pages(browser, sleep_val=1):
    """
    Gets the total number of pages of MoonBoard problems
    """
    found_page = True
    current_page = get_current_page(browser)
    while found_page:
        page_elem = click_next_page(browser, current_page)
        time.sleep(sleep_val)
        if page_elem is None:
            break
        current_page += 1
    return current_page


def process_all_problems(browser, problems_dict):
    """
    For a given page, collect all problems' metadata
    """
    problems, data_ids = get_problems(browser)
    for i, problem in enumerate(problems):
        if data_ids[i] in problems_dict:
            continue
        problems_dict[data_ids[i]] = get_problem_meta(problem)

    return problems_dict


def process_all_pages(browser, save_path='', num_tries=20, sleep_val=1, num_pages=-1):
    """
    Processes all pages and saves results into a dictionary
    """
    # Load problems dict, if it exists
    problems_dict = {}
    if os.path.exists(save_path):
        problems_dict = load_pickle(save_path)

    page_cnt = 0
    found_page = True
    current_page = get_current_page(browser)
    while found_page:
        page_cnt += 1
        for i in range(num_tries):
            try:
                true_current_page = get_current_page(browser)
                if current_page != true_current_page:
                    print('New page not loaded yet!')
                    raise ValueError()
                problems_dict = process_all_problems(browser, problems_dict)
                print('Processed page: %s!' % current_page)
                break
            except:
                print('Failed to process problems on page ' + str(current_page))
                time.sleep(sleep_val)
                continue

        # Save intermediate result
        if save_path != '':
            save_pickle(problems_dict, save_path)

        # Click to next page until pages run out
        page_elem = None
        for i in range(num_tries):
            try:
                page_elem = click_next_page(browser, current_page)
                print('Clicked page %s' % (current_page + 1))
                time.sleep(1)
                break
            except:
                print('Failed to click page %s' % (current_page + 1))
                time.sleep(sleep_val)
                continue

        # If the end of pages is reached
        if page_elem is None or page_cnt == num_pages:
            break

        # Flip to next page
        current_page += 1
        

    return problems_dict


def scrape_problems(browser, problems_dict, holds_path, failed_dict, failed_path, num_tries=10, sleep_val=1):
    """
    Accesses each problem page and scrapes data
    """
    uids = sorted(problems_dict.keys())

    # Iterate through each problem
    for i, uid in enumerate(uids):
        print(str(i + 1) + ' / ' + str(len(uids)))
        tmp_problem = copy.deepcopy(problems_dict[uid])

        # Specific problem has already been processed
        if ('moves' in tmp_problem) and ('repeats' in tmp_problem):
            continue

        # Specific URL does not load
        if check_fail_url(browser, tmp_problem['url']):
            print(str(uid) + ' failed')
            failed_dict[uid] = tmp_problem['url']

            save_pickle(failed_dict, failed_path)
            continue
        else:
            change = False
            browser.get(tmp_problem['url'])  # Navigate to page

            for x in range(num_tries):
                try:
                    if 'moves' not in tmp_problem:
                        target_text = get_target_script(browser)
                        json_dict = parse_target_json(target_text)
                        tmp_problem = parse_moves_and_others(json_dict, tmp_problem)
                        #tmp_problem['moves'] = parse_moves(json_dict)
                        change = True
                        
                except:
                    print('Failed to find moves / repeats')
                    time.sleep(sleep_val)
                    continue

            if change:
                problems_dict[uid] = tmp_problem
                save_pickle(problems_dict, holds_path)

    return problems_dict, failed_dict


# ----------------------------------------------------------------------------------------------------------------------
# Schema formatting functions
# ----------------------------------------------------------------------------------------------------------------------
def get_pos_map():
    """
    Defines a mapping from MoonBoard columns (designated A - K) to numerical values (0 - 10)
    """
    letters = string.ascii_uppercase[:11]
    return {l: i for i, l in enumerate(letters)}


def get_grade_map():
    """
    Defines a mapping of Fontainebleau grades to integer values
    """
    grade_map = {
        '6A': 0,
        '6A+': 1,
        '6B': 2,
        '6B+': 3,
        '6C': 4,
        '6C+': 5,
        '7A': 6,
        '7A+': 7,
        '7B': 8,
        '7B+': 9,
        '7C': 10,
        '7C+': 11,
        '8A': 12,
        '8A+': 13,
        '8B': 14,
        '8B+': 15,
    }
    return grade_map


def get_moves(problem):
    """
    Parses 'moves' attribute in raw mined MoonBoard metadata

    Input(s):
    - problem (dict): Dictionary of raw mined data

    Output(s):
    Three (3) list of list of ints
    """
    pos_map = get_pos_map()

    # Instantiate indexes
    start_idxs = []
    mid_idxs = []
    end_idxs = []

    if 'moves' in problem:
        moves = problem['moves']
        for m in moves:
            position = m['Description']
            position = [int(pos_map[position[0]]), int(position[1:])-1]  # Column index, row index
            if m['IsStart']:
                start_idxs.append(position)
            elif m['IsEnd']:
                end_idxs.append(position)
            else:
                mid_idxs.append(position)

    return start_idxs, mid_idxs, end_idxs


def process_raw_to_schema_basic(raw_dict):
    """
    Processes raw mined MoonBoard metadata into a basic format for neural network processing

    Input(s):
    - raw_dict (dict): Dictionary sample from [problems_dict_holds.pickle]
    """
    # Get move indexes
    start_idxs, mid_idxs, end_idxs = get_moves(raw_dict)

    # Get grade map
    grade_map = get_grade_map()

    # Instantiate basic dictionary
    basic_dict = {
        'url': raw_dict['url'],
        'start': start_idxs,
        'mid': mid_idxs,
        'end': end_idxs,
        #'grade': grade_map[raw_dict['grade']],
        'grade': raw_dict['grade'],
        'user_grade': raw_dict['UserGrade'],
        'is_benchmark': raw_dict['isBenchmark'],
        'repeats': raw_dict['repeats'],
        'problem_type': raw_dict['ProblemType'],
        'is_master': raw_dict['IsMaster'],
        'setter': raw_dict['setter'],
    }
    return basic_dict


def cast_to_basic_schema(raw_data):
    """
    Casts raw mined dictionary into a basic schema format, wrapper for process_raw_to_schema_basic()
    """
    formatted_data = dict()

    for problem_name, raw_problem in raw_data.items():
        try:
            formatted_data[problem_name] = process_raw_to_schema_basic(raw_problem)
        except:
            print('Failed to read %s' % problem_name)

    return formatted_data
