from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from oauth2client.tools import argparser

DEVELOPER_KEY = "AIzaSyArFQcyDH4e8DJYp9-CambiE5cZwQdUcvw"
YOUTUBE_API_SERVICE_NAME = "youtube"
YOUTUBE_API_VERSION = "v3"


def video_search(q, max_results=50, order="relevance", token=None, location=None, location_radius=None):

  youtube = build(YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION,developerKey=DEVELOPER_KEY)

  search_response = youtube.search().list(q=q,
                                          type="video",
                                          pageToken=token, #First page has token = None
                                          order = order,
                                          part="id,snippet",
                                          maxResults=max_results,
                                          location=location,
                                          locationRadius=location_radius,
                                          publishedAfter="2020-01-01T00:00:00Z",
                                          publishedBefore="2021-01-01T00:00:00Z",
                                          ).execute()
  
  total_num_results=search_response["pageInfo"]["totalResults"]
  print("Your search query resulted in a total of " + str(total_num_results) + " videos")
  
  videos = []

  for search_result in search_response.get("items", []):
    if search_result["id"]["kind"] == "youtube#video":
      videos.append(search_result)
  try:
      nexttok = search_response["nextPageToken"]
      return(nexttok, videos, total_num_results)
  except Exception as e:
      nexttok = "last_page"
      return(nexttok, videos, total_num_results)


def geo_query(video_id):
    youtube = build(YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION,developerKey=DEVELOPER_KEY)

    video_response = youtube.videos().list(id=video_id,
                                          part='snippet, recordingDetails, statistics',
                                          ).execute()

    return video_response


def video_comments_query(video_id, token=None):
    youtube = build(YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION,developerKey=DEVELOPER_KEY)

    response = youtube.commentThreads().list(videoId=video_id,
                                            pageToken=token,
                                            part='id,snippet,replies',
                                            order='time',
                                            maxResults=100, # Acceptable values are 1 to 100, inclusive. The default value is 20.
                                            #searchTerms='',
                                            textFormat="plainText",
                                            ).execute()

    tl_comments = []
    replies = []
  
    for comment_response_result in response.get("items", []):
      if comment_response_result["kind"] == "youtube#commentThread":
        tl_comments.append(comment_response_result["snippet"]["topLevelComment"])
        if comment_response_result["snippet"]["totalReplyCount"] > 0:  
          replies.extend(comment_response_result["replies"]["comments"]) 
        
    try:
        nexttok = response["nextPageToken"]
        return(nexttok, tl_comments, replies)
    except Exception as e:
        nexttok = "last_page"
        return(nexttok, tl_comments, replies)


