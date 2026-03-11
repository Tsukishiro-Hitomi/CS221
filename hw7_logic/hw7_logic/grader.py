#!/usr/bin/env python3

############################################################
# check python version
import sys
import warnings

if not (sys.version_info[0] == 3 and sys.version_info[1] == 12):
    warnings.warn(
        f"Note that you are not using python 3.12. Your code may not work in gradescope."
    )
############################################################

from logic import *

import pickle, gzip, os, random
import grader_util

grader = grader_util.Grader()
submission = grader.load("submission")


# name: name of this formula (used to load the models)
# predForm: the formula predicted in the submission
# preconditionForm: only consider models such that preconditionForm is true
def check_formula(name, predForm, preconditionForm=None, handle_grader=True):
    filename = os.path.join("models", name + ".pklz")
    objects, targetModels = pickle.load(gzip.open(filename))
    # If preconditionion exists, change the formula to
    preconditionPredForm = (
        And(preconditionForm, predForm) if preconditionForm else predForm
    )
    predModels = perform_model_checking(
        [preconditionPredForm], findAll=True, objects=objects
    )
    ok = True

    def hashkey(model):
        return tuple(sorted(str(atom) for atom in model))

    targetModelSet = set(hashkey(model) for model in targetModels)
    predModelSet = set(hashkey(model) for model in predModels)
    for model in targetModels:
        if hashkey(model) not in predModelSet:
            if handle_grader:
                grader.fail(
                    "Your formula (%s) fails to derive the following model that should be TRUE. The following printed statements are the facts given to the model:"
                    % predForm
                )
            ok = False
            print_model(model)
            return ok
    for model in predModels:
        if hashkey(model) not in targetModelSet:
            if handle_grader:
                grader.fail(
                    "Your formula (%s) derives the following model that should actually be FALSE. The following printed statements are the facts given to the model:"
                    % predForm
                )
            ok = False
            print_model(model)
            return ok
    if handle_grader:
        grader.add_message("You matched the %d models" % len(targetModels))
        grader.add_message("Example model: %s" % rstr(random.choice(targetModels)))
        grader.assign_full_credit()
    return ok


# name: name of this formula set (used to load the models)
# predForms: formulas predicted in the submission
# predQuery: query formula predicted in the submission
def add_parts(name, numForms, predictionFunc):
    # part is either an individual formula (0:numForms), all (combine everything)
    def check(part):
        predForms, predQuery = predictionFunc()
        # Don't require all parts to check one part
        minForms = part + 1 if type(part) == int else numForms
        if len(predForms) < minForms:
            grader.fail(
                "Wanted %d formulas, but got %d formulas:" % (numForms, len(predForms))
            )
            for form in predForms:
                print(("-", form))
            return
        if part == "all":
            check_formula(name + "-all", AndList(predForms))
        elif part == "run":
            # Actually run it on a knowledge base
            # kb = create_resolution_kb()  # Too slow!
            kb = create_model_checking_kb()

            # Need to tell the KB about the objects to do model checking
            filename = os.path.join("models", name + "-all.pklz")
            objects, targetModels = pickle.load(gzip.open(filename))
            for obj in objects:
                kb.tell(Atom("Object", obj))

            # Add the formulas
            for predForm in predForms:
                response = kb.tell(predForm)
                show_kb_response(response)
                grader.require_is_true(response.status in [CONTINGENT, ENTAILMENT])
            response = kb.ask(predQuery)
            show_kb_response(response)

        else:  # Check the part-th formula
            check_formula(name + "-" + str(part), predForms[part])

    def createCheck(part):
        return lambda: check(part)  # To create closure

    # run part-all once first for combined correctness, if true, trivially assign full score for all subparts
    # this is to account for student adding formulas to the list in different orders but still get
    # the combined preds correct.
    all_is_correct = False
    try:
        predForms, predQuery = predictionFunc()
        all_is_correct = check_formula(
            name + "-all", AndList(predForms), handle_grader=False
        )
    except BaseException:
        pass

    for part in list(range(numForms)) + ["all", "run"]:
        if part == "all":
            description = "test implementation of %s for %s" % (part, name)
        elif part == "run":
            description = "test implementation of %s for %s" % (part, name)
        else:
            description = "test implementation of statement %s for %s" % (part, name)
        if all_is_correct and not part in ["all", "run"]:
            grader.add_basic_part(
                name + "-" + str(part),
                lambda: grader.assign_full_credit(),
                max_points=1,
                max_seconds=10000,
                description=description,
            )
        else:
            grader.add_basic_part(
                name + "-" + str(part),
                createCheck(part),
                max_points=1,
                max_seconds=10000,
                description=description,
            )


############################################################
# Problem 1: propositional logic

grader.add_basic_part(
    "1a",
    lambda: check_formula("1a", submission.formula1a()),
    max_points=2,
    description="Test formula 1a implementation",
)
grader.add_basic_part(
    "1b",
    lambda: check_formula("1b", submission.formula1b()),
    max_points=2,
    description="Test formula 1b implementation",
)
grader.add_basic_part(
    "1c",
    lambda: check_formula("1c", submission.formula1c()),
    max_points=2,
    description="Test formula 1c implementation",
)

############################################################
# Problem 2: first-order logic

formula2a_precondition = AntiReflexive("Parent")
formula2b_precondition = AntiReflexive("Child")
formula2c_precondition = AntiReflexive("Child")
formula2d_precondition = AntiReflexive("Parent")
grader.add_basic_part(
    "2a",
    lambda: check_formula("2a", submission.formula2a(), formula2a_precondition),
    max_points=2,
    description="Test formula 2a implementation",
)
grader.add_basic_part(
    "2b",
    lambda: check_formula("2b", submission.formula2b(), formula2b_precondition),
    max_points=2,
    description="Test formula 2b implementation",
)
grader.add_basic_part(
    "2c",
    lambda: check_formula("2c", submission.formula2c(), formula2c_precondition),
    max_points=2,
    description="Test formula 2c implementation",
)
grader.add_basic_part(
    "2d",
    lambda: check_formula("2d", submission.formula2d(), formula2d_precondition),
    max_points=2,
    description="Test formula 2d implementation",
)

############################################################
# Problem 3: liar puzzle

# Add 3a-[0-5], 3a-all, 3a-run
add_parts("3a", 6, submission.liar)

############################################################
# Problem 4: odd and even integers

# Add 5a-[0-5], 5a-all, 5a-run
add_parts("4a", 6, submission.ints)

############################################################
# Problem 5: semantic parsing

import nlparser


def get_top_derivation(sentence, languageProcessor, grammar):
    # Return (action, formula)
    # - action is either |tell| or |ask|
    # - formula is a logical form
    print()
    print((">>>", sentence))
    utterance = nlparser.Utterance(sentence, languageProcessor)
    print(("Utterance:", utterance))
    derivations = nlparser.parse_utterance(utterance, grammar, verbose=0)
    if not derivations:
        raise Exception("Error: Parsing failed. (0 derivations)")
    return derivations[0].form


def test_knowledge_base(examples, ruleCreator):
    # Test the logical forms by querying the knowledge base.
    kb = create_model_checking_kb()
    # kb = create_resolution_kb()
    languageProcessor = nlparser.create_base_language_processor()
    # Need to tell kb about objects
    for obj in nlparser.BASE_OBJECTS:
        kb.tell(Atom("Object", obj.lower()))

    # Parse!
    grammar = nlparser.create_base_english_grammar() + [ruleCreator()]
    for sentence, expectedResult in examples:
        mode, formula = get_top_derivation(sentence, languageProcessor, grammar)
        print(("The parser returns:", (mode, formula)))
        grader.require_is_equal(expectedResult[0], mode)
        if mode == "tell":
            response = kb.tell(formula)
        if mode == "ask":
            response = kb.ask(formula)
        print(("Knowledge base returns:", response))
        grader.require_is_equal(expectedResult[1], response.status)
        # kb.dump()


### 5a


def test_5a_1():
    examples = [
        ("Every person likes some cat.", ("tell", CONTINGENT)),
        ("Every cat is a mammal.", ("tell", CONTINGENT)),
        ("Every person likes some mammal?", ("ask", ENTAILMENT)),
    ]
    test_knowledge_base(examples, submission.create_rule1)


grader.add_basic_part(
    "5a-1",
    test_5a_1,
    max_points=0.5,
    extra_credit=True,
    max_seconds=60,
    description="Check basic behavior of rule",
)


def test_5a_2():
    examples = [
        ("Every person likes some cat.", ("tell", CONTINGENT)),
        ("Every tabby is a cat.", ("tell", CONTINGENT)),
        ("Every person likes some tabby?", ("ask", CONTINGENT)),
    ]
    test_knowledge_base(examples, submission.create_rule1)


grader.add_basic_part(
    "5a-2",
    test_5a_2,
    max_points=0.5,
    extra_credit=True,
    max_seconds=60,
    description="Check basic behavior of rule",
)


def test_5a_3():
    examples = [
        ("Every person likes some cat.", ("tell", CONTINGENT)),
        ("Every person is a mammal.", ("tell", CONTINGENT)),
        ("Every mammal likes some cat?", ("ask", CONTINGENT)),
    ]
    test_knowledge_base(examples, submission.create_rule1)


grader.add_basic_part(
    "5a-3",
    test_5a_3,
    max_points=0.5,
    extra_credit=True,
    max_seconds=60,
    description="Check basic behavior of rule",
)


def test_5a_4():
    examples = [
        ("Every person likes some cat.", ("tell", CONTINGENT)),
        ("Garfield is a cat.", ("tell", CONTINGENT)),
        ("Jon is a person.", ("tell", CONTINGENT)),
        ("Jon likes Garfield?", ("ask", CONTINGENT)),
    ]
    test_knowledge_base(examples, submission.create_rule1)


grader.add_basic_part(
    "5a-4",
    test_5a_4,
    max_points=0.5,
    extra_credit=True,
    max_seconds=60,
    description="Check basic behavior of rule",
)


def test_5a_5():
    examples = [
        ("Every person likes some cat.", ("tell", CONTINGENT)),
        ("Every tabby is a cat.", ("tell", CONTINGENT)),
        ("Garfield is a tabby.", ("tell", CONTINGENT)),
        ("Jon is a person.", ("tell", CONTINGENT)),
        ("Jon likes Garfield?", ("ask", CONTINGENT)),
        ("Every person likes some tabby?", ("ask", CONTINGENT)),
    ]
    test_knowledge_base(examples, submission.create_rule1)


grader.add_basic_part(
    "5a-5",
    test_5a_5,
    max_points=0.5,
    extra_credit=True,
    max_seconds=60,
    description="Check basic behavior of rule",
)

### 5b


def test_5b_1():
    examples = [
        ("There is some cat that every person likes.", ("tell", CONTINGENT)),
        ("Every cat is a mammal.", ("tell", CONTINGENT)),
        ("There is some mammal that every person likes?", ("ask", ENTAILMENT)),
    ]
    test_knowledge_base(examples, submission.create_rule2)


grader.add_basic_part(
    "5b-1",
    test_5b_1,
    max_points=0.5,
    extra_credit=True,
    max_seconds=60,
    description="Check basic behavior of rule",
)


def test_5b_2():
    examples = [
        ("There is some cat that every person likes.", ("tell", CONTINGENT)),
        ("Jon is a person.", ("tell", CONTINGENT)),
        ("Jon likes some cat?", ("ask", ENTAILMENT)),
    ]
    test_knowledge_base(examples, submission.create_rule2)


grader.add_basic_part(
    "5b-2",
    test_5b_2,
    max_points=0.5,
    extra_credit=True,
    max_seconds=60,
    description="Check basic behavior of rule",
)


def test_5b_3():
    examples = [
        ("There is some cat that every person likes.", ("tell", CONTINGENT)),
        ("Garfield is a cat.", ("tell", CONTINGENT)),
        ("Jon is a person.", ("tell", CONTINGENT)),
        ("Jon likes Garfield?", ("ask", CONTINGENT)),
    ]
    test_knowledge_base(examples, submission.create_rule2)


grader.add_basic_part(
    "5b-3",
    test_5b_3,
    max_points=0.5,
    extra_credit=True,
    max_seconds=60,
    description="Check basic behavior of rule",
)

### 5c


def test_5c_1():
    examples = [
        (
            "If a person likes a cat then the former feeds the latter.",
            ("tell", CONTINGENT),
        ),
        ("Jon is a person.", ("tell", CONTINGENT)),
        ("Jon likes Garfield.", ("tell", CONTINGENT)),
        ("Jon feeds Garfield?", ("ask", CONTINGENT)),
    ]
    test_knowledge_base(examples, submission.create_rule3)


grader.add_basic_part(
    "5c-1",
    test_5c_1,
    max_points=0.5,
    extra_credit=True,
    max_seconds=60,
    description="Check basic behavior of rule",
)


def test_5c_2():
    examples = [
        (
            "If a person likes a cat then the former feeds the latter.",
            ("tell", CONTINGENT),
        ),
        ("Jon is a person.", ("tell", CONTINGENT)),
        ("Jon likes Garfield.", ("tell", CONTINGENT)),
        ("Garfield is a cat.", ("tell", CONTINGENT)),
        ("Jon feeds Garfield?", ("ask", ENTAILMENT)),
    ]
    test_knowledge_base(examples, submission.create_rule3)


grader.add_basic_part(
    "5c-2",
    test_5c_2,
    max_points=0.5,
    extra_credit=True,
    max_seconds=60,
    description="Check basic behavior of rule",
)


def test_5c_3():
    examples = [
        (
            "If a person likes a cat then the former feeds the latter.",
            ("tell", CONTINGENT),
        ),
        ("Jon likes Garfield.", ("tell", CONTINGENT)),
        ("Garfield is a cat.", ("tell", CONTINGENT)),
        ("Jon feeds Garfield?", ("ask", CONTINGENT)),
    ]
    test_knowledge_base(examples, submission.create_rule3)


grader.add_basic_part(
    "5c-3",
    test_5c_3,
    max_points=0.5,
    extra_credit=True,
    max_seconds=60,
    description="Check basic behavior of rule",
)


def test_5c_4():
    examples = [
        (
            "If a person likes a cat then the former feeds the latter.",
            ("tell", CONTINGENT),
        ),
        ("Jon is a person.", ("tell", CONTINGENT)),
        ("Jon likes some cat.", ("tell", CONTINGENT)),
        ("Garfield is a cat.", ("tell", CONTINGENT)),
        ("Jon feeds Garfield?", ("ask", CONTINGENT)),
    ]
    test_knowledge_base(examples, submission.create_rule3)


grader.add_basic_part(
    "5c-4",
    test_5c_4,
    max_points=0.5,
    extra_credit=True,
    max_seconds=60,
    description="Check basic behavior of rule",
)

############################################################
# Problem 6: explainability of logic-based systems
grader.add_manual_part(
    "6a", max_points=2, description="Explanation of logic-based systems"
)
grader.add_manual_part(
    "6b", max_points=2, description="Why explanation might not be adequate"
)
grader.add_manual_part(
    "6c", max_points=2, description="Four considerations of explainability"
)

############################################################
# Problem 7: Applications of Soundness and Completeness in AI Systems
grader.add_manual_part("7a", max_points=4, description="LLM soundness and completeness")
grader.add_manual_part(
    "7b",
    max_points=2,
    description="First part of comparing two systems based on soundness and completeness",
)
grader.add_manual_part(
    "7c",
    max_points=2,
    description="Second part of comparing two systems based on soundness and completeness",
)


grader.grade()
