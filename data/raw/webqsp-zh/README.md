# WebQSP-zh
WebQSP-zh consists of 4,737 questions mannually translated from WebQSP by Chinese native speakers. Among them, 3,098 questions are for training and 1,639 questions are for test.

## Format
```
{
    "qid": "wqs000000",
    "answers": [
      "Jamaican English",
      "Jamaican Creole English Language"
    ],
    "question": "牙买加人说什么？",
    "ner": [
      [
        "jamaican",
        "MISC"
      ],
      ...
    ],
    "freebase_key": "jamaica",
    "rel_path": [
      [
        [
          "/location/country/languages_spoken"
        ],
        2
      ],
      ...
    ],
    "webqsp_info": {
      "QuestionId": "WebQTest-0",
      ...
    },
    "original_question": "what does jamaican people speak?",
    "question_type": "zh"
}
```

* `ner`, `freebase_key` and `rel_path` are the NER results, topic entities in the Freebase, and relation paths, extracted from [brmson/dataet-factoid-webquestions](https://github.com/brmson/dataset-factoid-webquestions).
* `webqsp_info` is the information from the original WebQSP dataset.