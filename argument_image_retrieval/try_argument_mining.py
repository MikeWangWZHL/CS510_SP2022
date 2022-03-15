from canary.argument_pipeline import download_model, load_model, analyse_file
import spacy
'''demo'''
# if __name__ == "__main__":
#     # Download pretrained models from the web (unless you fancy creating them yourself)
#     # Training the models takes a while so I'd advise against it.
#     download_model("all")

#     # load the detector
#     detector = load_model("argument_detector")

#     # outputs false
#     print(detector.predict("cats are pretty lazy animals"))

#     # outputs true
#     print(detector.predict(
#         "If a criminal knows that a person has a gun , they are much less likely to attempt a crime ."))


if __name__ == '__main__':
    file_path = '/shared/nas/data/m1/wangz3/cs510_sp2022/argument_image_retrieval/dataset/images/I00/I00a534edc86bf5cd/pages/P7e54e192ebcb96db/snapshot/text.txt'
    
    detector = load_model("argument_detector")
    nlp = spacy.load("en_core_web_sm", disable=['ner','tagger','lemmatizer'])
    
    threshold = 0.95

    argument_sentences = []
    with open(file_path, 'r') as f:
        print('detecting...')
        for line in f:
            if len(line) > 10:
                doc = nlp(line)
                for sent in doc.sents:
                    if len(sent.text) > 10:
                        pred = detector.predict(sent.text, probability=True)
                        if pred[True] >= threshold:
                            argument_sentences.append(sent.text)
    
    for sent in argument_sentences:
        print(sent)