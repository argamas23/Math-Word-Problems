# coding: utf-8

from copy import deepcopy
import re


class Et:
    """
    Expression Tree node class for constructing expression trees
    from postfix expressions
    """
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None


def construct_exp_tree(postfix):
    """
    Builds an expression tree from a postfix notation expression
    
    Args:
        postfix: List of tokens in postfix notation
        
    Returns:
        Root node of the expression tree
    """
    stack = []

    # Traverse through every token of input expression
    for char in postfix:
        # If operand, create a leaf node and push into stack
        if char not in ["+", "-", "*", "/", "^"]:
            t = Et(char)
            stack.append(t)
        # If operator, create an operator node with two children
        else:
            t = Et(char)
            t1 = stack.pop()  # Right child
            t2 = stack.pop()  # Left child

            # Make them children
            t.right = t1
            t.left = t2

            # Add this subexpression to stack
            stack.append(t)
    
    # The last element in stack is the root of expression tree
    t = stack.pop()
    return t


def from_infix_to_postfix(expression):
    """
    Converts infix notation to postfix notation using stack
    
    Args:
        expression: List of tokens in infix notation
        
    Returns:
        List of tokens in postfix notation
    """
    st = list()  # Stack for operators
    res = list()  # Result list for postfix expression
    
    # Define operator precedence
    priority = {"+": 0, "-": 0, "*": 1, "/": 1, "^": 2}
    
    for e in expression:
        if e in ["(", "["]:
            st.append(e)
        elif e == ")":
            c = st.pop()
            while c != "(":
                res.append(c)
                c = st.pop()
        elif e == "]":
            c = st.pop()
            while c != "[":
                res.append(c)
                c = st.pop()
        elif e in priority:
            # Pop operators with higher or equal precedence
            while len(st) > 0 and st[-1] not in ["(", "["] and priority[e] <= priority[st[-1]]:
                res.append(st.pop())
            st.append(e)
        else:
            # Token is an operand
            res.append(e)
            
    # Pop any remaining operators
    while len(st) > 0:
        res.append(st.pop())
        
    return res


def from_infix_to_prefix(expression):
    """
    Converts infix notation to prefix notation
    
    Args:
        expression: List of tokens in infix notation
        
    Returns:
        List of tokens in prefix notation
    """
    st = list()  # Stack for operators
    res = list()  # Result list for prefix expression
    
    # Define operator precedence
    priority = {"+": 0, "-": 0, "*": 1, "/": 1, "^": 2}
    
    # Reverse the expression for prefix conversion
    expression = deepcopy(expression)
    expression.reverse()
    
    for e in expression:
        if e in [")", "]"]:
            st.append(e)
        elif e == "(":
            c = st.pop()
            while c != ")":
                res.append(c)
                c = st.pop()
        elif e == "[":
            c = st.pop()
            while c != "]":
                res.append(c)
                c = st.pop()
        elif e in priority:
            # Use strictly less-than in prefix conversion
            while len(st) > 0 and st[-1] not in [")", "]"] and priority[e] < priority[st[-1]]:
                res.append(st.pop())
            st.append(e)
        else:
            # Token is an operand
            res.append(e)
            
    # Pop any remaining operators
    while len(st) > 0:
        res.append(st.pop())
        
    # Reverse the result to get prefix notation
    res.reverse()
    return res


def out_expression_list(test, output_lang, num_list, num_stack=None):
    """
    Converts model output indices to human-readable expressions
    
    Args:
        test: List of token indices
        output_lang: Language object containing index-to-token mapping
        num_list: List of numeric values
        num_stack: Stack of numeric positions
        
    Returns:
        List of expression tokens
    """
    max_index = output_lang.n_words
    res = []
    for i in test:
        if i < max_index - 1:
            idx = output_lang.index2word[i]
            if idx[0] == "N":
                if int(idx[1:]) >= len(num_list):
                    return None
                res.append(num_list[int(idx[1:])])
            else:
                res.append(idx)
        else:
            # Handle copied numbers
            pos_list = num_stack.pop()
            c = num_list[pos_list[0]]
            res.append(c)
    return res


def compute_postfix_expression(post_fix):
    """
    Evaluates a postfix expression
    
    Args:
        post_fix: List of tokens in postfix notation
        
    Returns:
        Result of expression evaluation or None if invalid
    """
    st = list()  # Stack for operands
    operators = ["+", "-", "^", "*", "/"]
    
    for p in post_fix:
        if p not in operators:
            # Handle special number formats
            pos = re.search(r"\d+\(", p)
            if pos:
                st.append(eval(p[pos.start(): pos.end() - 1] + "+" + p[pos.end() - 1:]))
            elif p[-1] == "%":
                st.append(float(p[:-1]) / 100)
            else:
                st.append(eval(p))
        elif p == "+" and len(st) > 1:
            a = st.pop()
            b = st.pop()
            st.append(a + b)
        elif p == "*" and len(st) > 1:
            a = st.pop()
            b = st.pop()
            st.append(a * b)
        elif p == "/" and len(st) > 1:
            a = st.pop()
            b = st.pop()
            if a == 0:
                return None
            st.append(b / a)
        elif p == "-" and len(st) > 1:
            a = st.pop()
            b = st.pop()
            st.append(b - a)
        elif p == "^" and len(st) > 1:
            a = st.pop()
            b = st.pop()
            st.append(a ** b)
        else:
            return None
            
    if len(st) == 1:
        return st.pop()
    return None


def compute_prefix_expression(pre_fix):
    """
    Evaluates a prefix expression
    
    Args:
        pre_fix: List of tokens in prefix notation
        
    Returns:
        Result of expression evaluation or None if invalid
    """
    st = list()  # Stack for operands
    operators = ["+", "-", "^", "*", "/"]
    
    # Reverse the prefix expression for evaluation
    pre_fix = deepcopy(pre_fix)
    pre_fix.reverse()
    
    for p in pre_fix:
        if p not in operators:
            # Handle special number formats
            pos = re.search(r"\d+\(", p)
            if pos:
                st.append(eval(p[pos.start(): pos.end() - 1] + "+" + p[pos.end() - 1:]))
            elif p[-1] == "%":
                st.append(float(p[:-1]) / 100)
            else:
                st.append(eval(p))
        elif p == "+" and len(st) > 1:
            a = st.pop()
            b = st.pop()
            st.append(a + b)
        elif p == "*" and len(st) > 1:
            a = st.pop()
            b = st.pop()
            st.append(a * b)
        elif p == "/" and len(st) > 1:
            a = st.pop()
            b = st.pop()
            if b == 0:
                return None
            st.append(a / b)
        elif p == "-" and len(st) > 1:
            a = st.pop()
            b = st.pop()
            st.append(a - b)
        elif p == "^" and len(st) > 1:
            a = st.pop()
            b = st.pop()
            if float(eval(b)) != 2.0 or float(eval(b)) != 3.0:
                return None
            st.append(a ** b)
        else:
            return None
            
    if len(st) == 1:
        return st.pop()
    return None