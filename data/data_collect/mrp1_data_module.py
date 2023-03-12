
from __future__ import annotations
import os
from typing import Union, List
import time
from datetime import datetime
from itertools import chain

from dataclasses import dataclass, field, fields, InitVar, asdict

#property, classmethod, property (incl. setter, getter, deleter)

import numpy as np
import pandas as pd

import praw # Reddit API Wrapper
from trafilatura import fetch_url, extract, bare_extraction
from courlan import validate_url, check_url, clean_url


@dataclass(frozen=False)
class Comment:
    '''
    Defines a comment on an article
    '''
    comment_id: str
    article_id: str
    date: int # unix timestamp
    username: str
    from_where: str # 'r/[subreddit]', 'youtube', ...
    body: str
    level: int
    parent_comment_id: str
    replies: list[str] = field(default_factory=lambda: [], repr=False)
    replies_ids: list[str] = field(default_factory=lambda: [], repr=False)

    def __post_init__(self):
        pass

    def add_reply(self, reply):
        self.replies.append(reply)
        self.replies_ids.append(reply.comment_id)
            
@dataclass(frozen=False)
class Article:
    article_id: str #Article ID on the host website or post ID on e.g. Reddit
    article_type: str # 'web_article', 'video_description', ...
    link: str
    posted_by: str #Post author; only when applicable
    date_posted: int #Post date, unix timestamp; when applicable
    post_title: str #Post title; when applicable
    post_body: str #Post body; when applicable
    date: str
    author: str 
    headline: str
    body: str
    char_count: int = field(init=False, default=None, repr=False)
    comments: list[Comment] = field(default_factory=lambda: [], repr=False)
    comments_ids: list[str] = field(default_factory=lambda: [], repr=False)
    
    def __post_init__(self):
        self.char_count = len(self.body)
    
    def add_comment(self, comment: Comment):
        self.comments.append(comment)
        self.comments_ids.append(comment.comment_id)

@dataclass(frozen=False)
class ArticleSet:
    label: str
    as_list: list[Article] = field(default_factory=lambda: [], repr=False)
    article_id_list: list[int] = field(default_factory=lambda: [], repr=False)
    
    def post_init(self):
        pass
    
    def add_article(self, article):
        self.as_list.append(article)
        self.article_id_list.append(article.article_id)
    
    def to_dataframes(self):
        article_list = []
        comment_list = []
        for art in self.as_list:
            art_dict = asdict(art)
            cmt_queue = art_dict["comments"]
            while len(cmt_queue) != 0:
                cmt_dict = cmt_queue.pop(0) #Remove the saved comment from the queue
                for reply_dict in cmt_dict["replies"]:
                    cmt_queue.append(reply_dict)
                
                del cmt_dict["replies"]
                comment_list.append(cmt_dict)
                    
            del art_dict["comments"]
            article_list.append(art_dict)  
            
        article_df = pd.DataFrame(article_list)
        comment_df = pd.DataFrame(comment_list)
        return article_df, comment_df
    
    @classmethod
    def from_dataframes(artset_cls, artset_descript, article_df, comment_df):
        artset = artset_cls(artset_descript=artset_descript, 
                comments=[], 
                comments_ids=[])
        
        for idx, df_row in article_df.iterrows(): #For each post/article
            
            art = Article(article_id=df_row["article_id"],
                        article_type=df_row["article_type"],
                        link=df_row["link"],
                        posted_by=df_row["posted_by"],
                        date_posted=df_row["date_posted"], 
                        post_title=df_row["post_title"],
                        post_body=df_row["post_body"],
                        date=df_row["date"],
                        author=df_row["author"], 
                        headline= df_row["headline"],
                        body=df_row["body"],
                        comments=[], 
                        comments_ids=df_row["comments_ids"])

            p_cmt_id_buffer, cmt_buffer = [], []
            art_comments_df = comment_df.loc[comment_df['parent_post_id'] == art.article_id]
            for level in reversed(range(0, np.max(comment_df['level']))): #For all levels
                comments = art_comments_df.loc[art_comments_df['level'] == level]
                next_p_cmt_id_buffer, next_cmt_buffer = [], [] #Save comments from this level and overwrite the main (old) buffers just after completing this level
                for cmt_row_idx in comments.index:
                    cmt = Comment(comment_id=comments.loc[cmt_row_idx, "comment_id"], 
                                    article_id=comments.loc[cmt_row_idx, "article_id"], 
                                    date=comments.loc[cmt_row_idx, "date"], 
                                    username=comments.loc[cmt_row_idx, "username"], 
                                    from_where=comments.loc[cmt_row_idx, "from_where"],
                                    body=comments.loc[cmt_row_idx, "body"],
                                    level=comments.loc[cmt_row_idx, "level"],
                                    parent_comment_id=comments.loc[cmt_row_idx, "parent_comment_id"], 
                                    replies=[], 
                                    replies_ids=comments.loc[cmt_row_idx, "replies_ids"])
                    
                    #Identity all replies to the current comment and store them in their parent
                    if cmt.comment_id in p_cmt_id_buffer:
                        for buffer_idx in p_cmt_id_buffer.index(cmt.comment_id):
                            cmt.add_reply(cmt_buffer[buffer_idx])

                    next_p_cmt_id_buffer.append(cmt.parent_comment_id)
                    next_cmt_buffer.append(cmt)
                    
                #Reset buffers and refill for next level
                p_cmt_id_buffer, cmt_buffer = next_p_cmt_id_buffer, next_cmt_buffer
            
            for cmt in cmt_buffer: #Add all top level comments to the article
                art.add_comment(cmt)
            
            artset.add_article(art)
            
        return artset
    
    def save_to_csv(self, filepath):
        '''
        Stores the current (self) articleset to a csv file 
        
        Parameters
        ----------
        filepath : str
            The path to the output csv file

        Returns
        -------
        
        None
        '''
        article_df, comment_df = self.to_dataframes()
        article_filename, comment_filename = filepath + '_articles.csv', filepath + '_comments.csv'
        article_df.to_csv(article_filename, sep=',', header=True, index=True, index_label=None)
        comment_df.to_csv(comment_filename, sep=',', header=True, index=True, index_label=None)
    	
    def load_from_csv(artset_cls, filepath, artset_descript):
        '''
        Loads an articleset from a csv file 
        
        Parameters
        ----------
        filepath : str
            The path to the input csv file

        Returns
        -------
        
        None
        '''
        article_filename, comment_filename = filepath + '_articles.csv', filepath + '_comments.csv'
        article_df = pd.read_csv(article_filename, sep=',', header=0, index_col=0)
        comment_df = pd.read_csv(comment_filename, sep=',', header=0, index_col=0)
        artset_cls.from_dataframes(artset_descript=artset_descript, article_df=article_df, comment_df=comment_df)
    
    @classmethod
    def from_reddit_dfs(artset_cls, artset_descript, post_df, comment_df, scraper):
        artset = artset_cls(label=artset_descript, 
                            as_list=[], 
                            article_id_list=[])
                
        for idx, df_row in post_df.iterrows(): #For each post/article
            
            #- - - - - - PARSE ARTICLE BODY 6 title: parse_article_from_link(url)
            parsed_article = scraper.parse_article_url(df_row["ext_link_url"])
            parsed_article["article_type"] = "posted_link"

            # - - - - - - - - 
            
            art = Article(article_id=df_row["id"],
                        article_type=parsed_article["article_type"], 
                        link=df_row["ext_link_url"],
                        posted_by=df_row["author_name"], #Note: This should be the author of the reddit submission/post, not the author of the article
                        date_posted=df_row["created_utc"],
                        post_title=df_row["title"], 
                        post_body=df_row["selftext"],
                        date=parsed_article["date"],
                        author=parsed_article["author"], #Note: This should be the author of the article, not the author of the reddit submission/post
                        headline=parsed_article["headline"],
                        body=parsed_article["body"],
                        comments=[], 
                        comments_ids=[])

            p_cmt_id_buffer, cmt_buffer = [], []
            art_comments_df = comment_df.loc[comment_df['parent_post_id'] == art.article_id]
            for level in reversed(range(0, np.max(comment_df['depth'].astype(int)))): #For all levels
                comments = art_comments_df.loc[art_comments_df['depth'] == level]
                next_p_cmt_id_buffer, next_cmt_buffer = [], [] #Save comments from this level and overwrite the main (old) buffers just after completing this level
                for cmt_row_idx in comments.index:
                    cmt = Comment(comment_id=comments.loc[cmt_row_idx, "id"], 
                                    article_id=comments.loc[cmt_row_idx, "parent_post_id"], 
                                    date=comments.loc[cmt_row_idx, "created_utc"], 
                                    username=comments.loc[cmt_row_idx, "author_name"], 
                                    from_where=comments.loc[cmt_row_idx, "subreddit_name_prefixed"],
                                    body=comments.loc[cmt_row_idx, "body"],
                                    level=comments.loc[cmt_row_idx, "depth"],
                                    parent_comment_id=comments.loc[cmt_row_idx, "parent_comment_id"], 
                                    replies=[], 
                                    replies_ids=[])
                    
                    #Identity all replies to the current comment and store them in their parent
                    if cmt.comment_id in p_cmt_id_buffer: 
                        for buffer_idx in p_cmt_id_buffer.index(cmt.comment_id):
                            cmt.add_reply(cmt_buffer[buffer_idx])

                    next_p_cmt_id_buffer.append(cmt.parent_comment_id)
                    next_cmt_buffer.append(cmt)
                    
                #Reset buffers and refill for next level
                p_cmt_id_buffer, cmt_buffer = next_p_cmt_id_buffer, next_cmt_buffer
            
            for cmt in cmt_buffer: #Add all top level comments to the article
                art.add_comment(cmt)
            
            artset.add_article(art)
            
        return artset


class Scraper:
    def __init__(self, reddit_api_wrapper="praw"):
        assert reddit_api_wrapper in ["praw", "pmaw"] #PRAW is the default wrapper, PMAW is a wrapper for the third party API Pushshift.io
        self.reddit_api_wrapper = reddit_api_wrapper
        
    def parse_article_url(self, url):
        #Define a return dictionary for the case of an invalid or (non-resolvable) redirect url
        art_dict = {"article_type": "web_article",
                "link": url,
                "headline": "Unknown / Invalid or redirect URL",
                "author":  "Unknown / Invalid or redirect URL",
                "date": "Unknown / Invalid or redirect URL",
                "body": "Unknown / Invalid or redirect URL",
                }
        
        if validate_url(url)[0] and (check_url(url)[-1] == "youtube.com" or check_url(url)[-1] == "youtu.be"): #Check whether url is a youtube video
            pass #TODO: Implement youtube video parsing via youtube API
        elif validate_url(url)[0]: #check whether url can be resolved or is damaged
            url = clean_url(url)
            downloaded = fetch_url(url)

            result = bare_extraction(downloaded, 
                                        url=url, 
                                        #include_comments=True,
                                        #include_formatting=False,
                                        favor_precision=True, #favor recall
                                        with_metadata=True,
                                        #deduplicate=True, 
                                        only_with_metadata=True,
                                        #url_blacklist=None,
                                        as_dict=True,
                                        )

            if result: #If the article was successfully parsed; e.g. no redirect url
                #Replace the return dictionary with the parsed data
                art_dict = {"article_type": "web_article",
                            "link": url,
                            "headline": result["title"],
                            "author": result["author"],
                            "date": result["date"],
                            "body": result["text"],
                            }
        return art_dict
    
    def query_from_subreddits(self,query, subreddits, post_limit_per_subreddit=1000, sort_by="relevance"):
        reddit = praw.Reddit(
                client_id="DIAteHPvd-6mPi8WMCykog",
                client_secret="PckY-T0xsvax4OgrWWvBkTxVIwvapg",
                #password="bonGux-gyrro0-zuvhuv",
                user_agent="python:DIAteHPvd-6mPi8WMCykog:0.1 scraper (by u/SciPhaM) ",
                #username="SciPhaM",
                )
        
        if self.reddit_api_wrapper == "praw": #PRAW Based Search and Retrieval
            assert post_limit_per_subreddit <= 1000 #PRAW has a limit of 1000 posts per query
            assert sort_by in ["relevance", "new", "comments"]
            
            # ----- Search and Retrieval ----- #

            combin_post_df, combin_comment_df = pd.DataFrame(), pd.DataFrame()
            
            query_result_list = []
            for subreddit in subreddits:
                print("Searching in subreddit: ", subreddit, "...")
                query_result = reddit.subreddit(subreddit).search(query=query,
                                                            sort=sort_by, # "comments",
                                                            limit=post_limit_per_subreddit,
                                                            time_filter="all",
                                                            )
                query_result_list.append(query_result)
            	
            query_result_concat = chain(*query_result_list)
            post_buffer = []
            comment_buffer = []

            cntr = 0
            
            for post_id in query_result_concat:
                cntr += 1
                
                start_time_subm = time.time() #%
                post = reddit.submission(id=post_id)
                if post.num_comments > 0:
                    start_time_postproc = time.time() #%
                    post_dict = post.__dict__
                    post_dict["author"] = post_dict["author"].name
                    post_buffer.append(post_dict)
                    print("--- %s seconds --- post processed" % (time.time() - start_time_postproc)) #%
                    
                    post.comments.replace_more(limit=None)
                    for comment in post.comments.list():
                        comment_dict = comment.__dict__
                        
                        #if comment_dict["body"] != "[removed]":  #Some comments were deleted, don't include them
                        comment_dict["_submission"] = comment_dict["_submission"].id
                        try:
                            comment_dict["author"] = comment_dict["author"].name
                        except: #If author or comment was deleted, don't include it in the data
                            continue
                        else:
                            comment_dict["_replies"] = tuple([reply.id for reply in comment_dict["_replies"]])
                    
                            comment_buffer.append(comment_dict)
                            
                    
                print("--- %s seconds --- submission %s total" % (time.time() - start_time_subm, cntr)) #%
                
            
            print(len(post_buffer))

            #Store as a DataFrames for convenience
            post_buffer = pd.DataFrame(post_buffer)
            comment_buffer = pd.DataFrame(comment_buffer)

            # ----- Post Processing ----- #
            #Only pass on the relveant features
            
            post_features = [ 'id', 'permalink', 'subreddit_id', 'subreddit_name_prefixed','author', 'author_fullname', 'title','created_utc', 'is_self','selftext','selftext_html' , 'url', 'media', 'num_comments']
            comment_features = ['id', 'permalink','subreddit_id','subreddit_name_prefixed', 'author', 'author_fullname','_submission', 'link_id', 'parent_id' ,'created_utc', 'body', 'body_html', 'depth' , '_replies']
            
            post_df = post_buffer[post_features]
            comment_df = comment_buffer[comment_features]

            post_df = post_df.rename(columns={'author': 'author_name', 'author_fullname': 'author_id', 'url': 'ext_link_url'}, inplace=False)
            comment_df = comment_df.rename(columns={'author': 'author_name', 'author_fullname': 'author_id','_submission': 'parent_post_id', 'parent_id': 'parent_comment_id', '_replies': 'replies_ids'}, inplace=False)

            post_df.loc[:,"subreddit_id"] = [item.split("_")[-1] for item in post_df.loc[:, "subreddit_id"]] #Remove the t5_ prefix
            post_df.loc[:,"author_id"] = [item.split("_")[-1] for item in post_df.loc[:, "author_id"]] #Remove the t2_ prefix

            comment_df.loc[:,"subreddit_id"] = [item.split("_")[-1] for item in comment_df.loc[:,"subreddit_id"]] #Remove the t5_ prefix
            comment_df.loc[:,"author_id"] = [item.split("_")[-1] for item in comment_df.loc[:, "author_id"]] #Remove the t2_ prefix

            comment_df.loc[:,"parent_comment_id"] = [item.split("_")[-1] if item.split("_")[0] == 't3' else None for item in comment_df.loc[:, "parent_comment_id"]] #Remove the t3_ prefix and substitue t1_ ids with None (for top level comments)

            combin_post_df = pd.concat([combin_post_df, post_df], ignore_index=True)
            combin_comment_df = pd.concat([combin_comment_df, comment_df], ignore_index=True)
            
                                
            return (combin_post_df, combin_comment_df)

        
        

def main():
    scraper = Scraper(reddit_api_wrapper="praw")
    post_df, comment_df = scraper.query_from_subreddits(query="quantum self:no",
                                                        subreddits=["science", "quantum"],     
                                                        post_limit_per_subreddit=20, 
                                                        sort_by="relevance",
                                                        )


    filepath = "../"
    #Convert to a standardized object structure before exporting to csv files:
    artset = ArticleSet.from_reddit_dfs(artset_descript="Test ArticleSet", 
                                        post_df = post_df, 
                                        comment_df = comment_df,
                                        scraper=scraper,)

    #print(artset.as_list[0].comments[2])

    artset.save_to_csv(filepath)

if __name__ == '__main__':
    main()
 