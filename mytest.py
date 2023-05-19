from deepchecks.nlp.utils.text_properties import calculate_builtin_properties
import pandas as pd
# tweet = pd.read_csv('./deepchecks/nlp/datasets/assets/tweet_emotion/tweet_emotion_data.csv')
# import time


# st = time.time()
# et = time.time()
text_data = [
    "Please send your inquiries to info@example.com or support@example.com. We are happy to assist you.",
    "Contact us at john.doe@example.com or jane.smith@example.com for further information. Looking forward to hearing from you.",
    "For any questions or concerns, email sales@example.com or reach out to customerservice@example.com. We're here to help.",
    "You can contact me directly at samantha.wilson@example.com or use the team email address marketing@example.com.",
    "If you have any feedback or suggestions, feel free to email us at feedback@example.com, support@example.com, feedback@example.com."
]

    # Act
result = calculate_builtin_properties(text_data, include_properties=['Count Email Address', 'Count Unique Email Address'])[0]
print(result['Count Email Address'], result['Count Unique Email Address'])
# print(et-st)
# import re
# url_pattern = r'https?://?(?:[-\w.]|(?:%[\da-fA-F]{2}))+'
# url_pattern = r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+'
# text = """"The results were found at http://www.google.com and it redirects to 
#         https://www.deepchecks.com and there we can find the links to all social medias such
#         as http://gmail.com, https://fb.com, https://www.deepchecks.com, https://www.google.com
#         """
# print(re.findall(url_pattern, text))