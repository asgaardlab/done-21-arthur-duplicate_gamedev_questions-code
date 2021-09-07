import os
import pandas as pd
from multiprocessing import Pool
from xml.etree.ElementTree import iterparse

# Local imports
from utils import paths, read, save
from utils.consts import n_procs

post_types = {"questions": 1, "answers": 2}


def unpack_gamedev_se():
    """Runs the bash script to unpack data from the 7z file
    and creates a Posts.7z and a PostLinks.7z file.
    """
    print("- Unpacking Game Dev. Stack Exchange data")
    data_dir = paths.raw_dir("gamedev_se")
    # redirects 7z output to keep terminal clean
    os.system(f"cd {data_dir} ; bash unpack_gamedev_se.sh > /dev/null")


def extract_posts(ds):
    """Decompresses the Posts.7z archive for a given dataset"""
    print("-- Decompressing Posts.7z")
    data_dir = paths.raw_dir(ds)
    file = data_dir / "Posts.7z"
    # redirects 7z output to keep terminal clean
    os.system(f"7z x {file} -o{data_dir} -y > /dev/null")


def extract_post_links(ds):
    """Decompresses the PostLinks.7z archive for a given dataset"""
    print("-- Decompressing PostLinks.7z")
    data_dir = paths.raw_dir(ds)
    file = data_dir / "PostLinks.7z"
    # redirects 7z output to keep terminal clean
    os.system(f"7z x {file} -o{data_dir} -y > /dev/null")


def select_post_type(ds, post_type):
    """Selects posts from a given type from Posts.xml for a given dataset
    Questions = post_type == 1
    Answers = post_type == 2
    """
    extract_posts(ds)
    posts_path = paths.posts_xml(ds)

    if post_type == post_types["questions"]:
        print("-- Selecting questions from Posts.xml")
        path = paths.questions_xml(ds)
    else:
        print("-- Selecting answers from Posts.xml")
        path = paths.answers_xml(ds)

    # select using grep for performance
    os.system(f"grep -F 'PostTypeId=\"{post_type}\"' {posts_path} > {path}")
    # deleted Posts.xml to save space
    # saves space but has to decompress the 7z archive twice
    posts_path.unlink()


def split_xml_file(ds, post_type):
    """Splits a XML file into chunks of 1MM lines to allow
    multiprocessing and to limit memory usage.
    """
    data_dir = paths.raw_dir(ds)

    if post_type == post_types["questions"]:
        print("-- Splitting questions into chunks")
        file_name = "questions"
        path = paths.questions_xml(ds)
    else:
        print("-- Splitting answers into chunks")
        file_name = "answers"
        path = paths.answers_xml(ds)

    # reduce number of lines to save memory during parsing
    os.system(f"split -l 1000000 {path} {data_dir}/{file_name}_")
    path.unlink()  # Remove original XML file to save space

    splits = list(paths.raw_dir(ds).glob(f"{file_name}_*"))

    print(f"-- {len(splits)} splits")

    # wrap file in tags for proper XML syntax
    for path in splits:
        os.system(f"sed -i '1s/^/<posts>\\n/' {path}")  # top tag
        os.system(f"echo '</posts>' >> {path}")  # bottom tag

    return splits


def parse_questions_xml(questions_path):
    """Parses XML files containing question data"""
    questions = []

    for _, node in iterparse(questions_path, events=("end",)):
        if node.tag == "row":  # ignore starting and ending <post> tags
            questions.append(
                {
                    "id": node.attrib.get("Id"),
                    "title": node.attrib.get("Title"),
                    "body": node.attrib.get("Body"),
                    "tags": node.attrib.get("Tags"),
                    "accepted_answer": node.attrib.get("AcceptedAnswerId"),
                    "n_answers": node.attrib.get("AnswerCount"),
                }
            )
        node.clear()

    questions = pd.DataFrame(questions)
    # select accepted answers to append to answer data later
    accepted_answers = questions[["accepted_answer"]].dropna()
    questions = questions[["id", "n_answers", "title", "body", "tags"]]

    questions = questions.drop_duplicates("id")
    questions["n_answers"] = questions["n_answers"].apply(int)

    # make the string of tags comma separated ("<tag1><tag2>" -> "tag1,tag2")
    split_tag = lambda s: s.replace("><", ",").replace("<", "").replace(">", "")
    questions.tags = questions.tags.apply(split_tag)

    return questions, accepted_answers


def question_parser(i, ds, questions_xml):
    """Function for one subprocess parsing XML question data
    i -> number of the worker
    """
    print(f"--- Worker {i} started")
    questions, acc_ids = parse_questions_xml(questions_xml)

    qids = questions[["id"]]  # save question IDs to select dup pairs later

    save(questions, paths.question_texts(ds, i))
    # Remove the raw XML file to save space
    questions_xml.unlink()

    return qids, acc_ids


def extract_questions_xml(ds):
    """Extract all questions from XML archives for a given dataset"""
    print("- Extracting questions")
    post_type = post_types["questions"]

    select_post_type(ds, post_type)
    splits = split_xml_file(ds, post_type)

    print("-- Parsing questions")

    # multiprocess to increase speed. Reduce n_procs to save on memory
    args = [(i, ds, s) for i, s in enumerate(splits)]
    with Pool(n_procs) as p:
        res = p.starmap(question_parser, args)

    # suffix is unnecessary if there is only one split (less than 1M questions)
    if len(splits) == 1:
        paths.question_texts(ds, 0).rename(paths.question_texts(ds))

    # save list of all question ids
    qids = [r[0] for r in res]
    qids = pd.concat(qids).reset_index(drop=True)
    qids = qids.drop_duplicates()
    save(qids, paths.all_question_ids(ds))

    # save list of all accepted answers
    acc_ids = [r[1] for r in res]
    acc_ids = pd.concat(acc_ids).reset_index(drop=True)
    acc_ids = acc_ids.drop_duplicates()
    acc_ids = acc_ids.rename(columns={"accepted_answer": "id"})
    acc_ids = acc_ids[["id"]]
    save(acc_ids, paths.accepted_answer_ids(ds))


def parse_postlinks_xml(ds):
    """Parse and select all duplicate relations from PostLinks.xml for a given dataset"""
    pairs = []
    links_xml = paths.post_links_xml(ds)

    for _, node in iterparse(links_xml, events=("end",)):
        # LinkTypeId for duplicate relation is 3
        if node.attrib.get("LinkTypeId") == "3":
            pairs.append(
                {
                    "dup_id": node.attrib.get("PostId"),
                    "main_id": node.attrib.get("RelatedPostId"),
                }
            )
        node.clear()

    pairs = pd.DataFrame(pairs)

    return pairs


def extract_dup_pairs_xml(ds):
    """Extract duplicate question pairs from XML archives for a given dataset"""
    print("- Extracting duplicate pairs")
    extract_post_links(ds)

    print("-- Parsing duplicate pairs")
    dup_pairs = parse_postlinks_xml(ds)

    # Select only the pairs that have both questions in the set of question IDs
    qids = read(paths.all_question_ids(ds)).id

    dup_in_qs = dup_pairs.dup_id.isin(qids)
    main_in_qs = dup_pairs.main_id.isin(qids)

    dup_pairs = dup_pairs[dup_in_qs & main_in_qs]
    dup_pairs = dup_pairs.drop_duplicates()

    save(dup_pairs, paths.dup_pairs(ds))

    # remove PostLinks.xml to save space
    paths.post_links_xml(ds).unlink()


def parse_answers_xml(ds, answers_path):
    """Parses XML files containing answer data"""
    qids = read(paths.all_question_ids(ds)).id
    acc_ids = read(paths.accepted_answer_ids(ds)).id

    answers = []

    for _, node in iterparse(answers_path, events=("end",)):
        if node.tag == "row":
            answers.append(
                {
                    "id": node.attrib.get("Id"),
                    "question_id": node.attrib.get("ParentId"),
                    "score": node.attrib.get("Score"),
                    "body": node.attrib.get("Body"),
                    "post_date": node.attrib.get("CreationDate"),
                }
            )
        node.clear()

    answers = pd.DataFrame(answers)

    # only select answers that have a question in the dataset
    answers = answers[answers.question_id.isin(qids)]

    answers["score"] = answers.score.apply(int)

    # mark answers as accepted
    answers.loc[answers.id.isin(acc_ids), "accepted"] = True
    answers["accepted"] = answers["accepted"].fillna(False)

    return answers


def answer_parser(i, ds, answers_xml):
    """Function for one subprocess parsing XML answer data
    i -> number of the worker
    """
    print(f"--- Worker {i} started")
    answers = parse_answers_xml(ds, answers_xml)

    save_path = paths.answer_texts(ds, i)

    save(answers, save_path)
    # Removes XML file to save space
    answers_xml.unlink()


def extract_answers_xml(ds):
    """Extract all questions from XML archives for a given dataset"""
    print("- Extracting answers")

    post_type = post_types["answers"]

    select_post_type(ds, post_type)
    splits = split_xml_file(ds, post_type)

    print("-- Parsing answers")

    # multiprocess to increase speed. Reduce n_procs to save on memory
    args = [(i, ds, s) for i, s in enumerate(splits)]
    with Pool(n_procs) as p:
        p.starmap(answer_parser, args)

    # suffix is unnecessary if there is only one split (less than 1M questions)
    if len(splits) == 1:
        paths.answer_texts(ds, 0).rename(paths.answer_texts(ds))

    # remove accepted answer IDs file as it won't be used anymore
    paths.accepted_answer_ids(ds).unlink()


def extract_xml(ds):
    """Extract questions, answers and dup pairs from an XML archive"""
    extract_questions_xml(ds)
    extract_dup_pairs_xml(ds)
    extract_answers_xml(ds)


def extract_xml_datasets(datasets):
    unpack_gamedev_se()
    for ds in datasets:
        print(f"Extracting data from {ds}")
        extract_xml(ds)


if __name__ == "__main__":
    print("Extracting XML data")
    extract_xml_datasets(["gamedev_se", "stackoverflow"])
