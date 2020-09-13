#!/usr/bin/env python3

from src.nltk_utils import *

class TestNltkUtils:

  def test_tokenize(self):
      result = tokenize("What's on the menu?")
      assert type(result) is list
      assert len(result) == 6


  def test_stem(self):
      result = [stem(w) for w in ["organize", "organizes", "organizing"]]
      assert "organ" in result
      assert len(result) == 3


  def test_bag_of_words(self):
      result = bag_of_words("a", ["a", "b"])
      assert any(result) and not all(result)
      assert len(result) == 2