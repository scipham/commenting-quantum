from __future__ import annotations
import hashlib
from typing import Union, List
import time
from datetime import datetime
from itertools import chain
from more_itertools import locate
import asyncio
import json

from dataclasses import dataclass, field, fields, InitVar, asdict

#property, classmethod, property (incl. setter, getter, deleter)

import numpy as np
import pandas as pd

import asyncpraw as praw # Reddit API Wrapper
#import praw
import youtube_api as yapi #Requires youtube_api.py in the same directory


from trafilatura import fetch_url, extract, bare_extraction
from courlan import validate_url, check_url, clean_url


@dataclass(frozen=False)
class Comment:
    """ Defines a comment on a post/article. """
    
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
        """ 
        Adds a reply to the comment. 
        
        Parameters
        --------------
        reply : Comment
            The reply to be added.
            
        Returns
        --------------
        None
        """
        self.replies.append(reply)
        self.replies_ids.append(reply.comment_id)
            
@dataclass(frozen=False)
class Article:
    """ Defines a post/article."""
    
    article_id: str #Article ID on the host website or post ID on e.g. Reddit
    article_type: str # 'web_article', 'video_description', ...
    from_where: str # 'r/[subreddit]', 'youtube', ...
    link: str
    post_id: list[str] #Post IDs; only when applicable
    posted_by: list[str] #Post authors; only when applicable
    date_posted: list[int] #Post dates, unix timestamp; when applicable
    post_title: list[str] #Post titles; when applicable
    post_body: list[str] #Post bodies; when applicable
    date: str
    author: str 
    headline: str
    body: str
    char_count: int = field(init=False, default=None, repr=False)
    comments: list[Comment] = field(default_factory=lambda: [], repr=False)
    comments_ids: list[str] = field(default_factory=lambda: [], repr=False)
    
    def __post_init__(self):
        """
        Post-initialization; Setting the character count of the article body.
        
        Parameters
        --------------
        None
        
        Returns
        --------------
        None
        """
        self.char_count = len(self.body)
    
    def add_comment(self, comment: Comment):
        """
        Adds a comment to the article.
        
        Parameters
        --------------
        comment : Comment
            The comment to be added. 
        
        Returns
        --------------
        None
        """
        self.comments.append(comment)
        self.comments_ids.append(comment.comment_id)

@dataclass(frozen=False)
class ArticleSet:
    """ Defines a collection of articles. """	
    
    label: str
    as_list: list[Article] = field(default_factory=lambda: [], repr=False)
    article_id_list: list[int] = field(default_factory=lambda: [], repr=False)
    
    def post_init(self):
        pass
    
    def add_article(self, article):
        """
        Adds an article to the article set.
        
        Parameters
        --------------
        article : Article
            The article to be added.
            
        Returns
        --------------
        None
        """
        self.as_list.append(article)
        self.article_id_list.append(article.article_id)
    
    def to_dataframes(self):
        """
        Converts an ArticleSet object to two (serial) pandas DataFrames.
        
        Parameters
        --------------
        None
        
        Returns
        --------------
        article_df : pd.DataFrame
            Contains the (meta)data of all articles in the ArticleSet.
        comment_df : pd.DataFrame
            Contains the (meta)data of all comments and replies in the ArticleSet.
        """
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
        """
        Creates/imports an ArticleSet object from two (serial) DataFrames; such as those created by the to_dataframes() method.
        
        Parameters
        --------------
        artset_descript : str
            A label/description of the data in the to-be-created ArticleSet.
        article_df : pd.DataFrame
            Contains the (meta)data of all articles in the ArticleSet.
        comment_df : pd.DataFrame
            Contains the (meta)data of all comments and replies in the ArticleSet.

        Returns
        --------------
        artset : ArticleSet
            The ArticleSet object created from the two DataFrames.
        """
        artset = artset_cls(artset_descript=artset_descript, 
                comments=[], 
                comments_ids=[])
        
        for idx, df_row in article_df.iterrows(): #For each post/article
            
            art = Article(article_id=df_row["article_id"],
                        article_type=df_row["article_type"],
                        from_where=df_row["from_where"],
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
        """
        Stores the current (self) articleset to a csv file 
        
        Parameters
        ----------
        filepath : str
            The (directorz) path to the output csv files

        Returns
        -------
        None
        """
        article_df, comment_df = self.to_dataframes()
        #article_filename, comment_filename = filepath + '_articles.csv', filepath + '_comments.csv'
        article_filename, comment_filename = filepath + '_articles.csv', filepath + '_comments.csv'
        article_df.to_csv(article_filename, sep=',', header=True, index=True, index_label=None)
        comment_df.to_csv(comment_filename, sep=',', header=True, index=True, index_label=None)
    	
    def load_from_csv(artset_cls, filepath, artset_descript):
        """
        Loads/creates an articleset from two csv files (one for articles and one for comments)
        
        Parameters
        ----------
        filepath : str
            The path to the directory containing the two input csv files; "_articles.csv" and "_comments.csv"

        Returns
        -------
        None
        """
        article_filename, comment_filename = filepath + '_articles.csv', filepath + '_comments.csv'
        article_df = pd.read_csv(article_filename, sep=',', header=0, index_col=0)
        comment_df = pd.read_csv(comment_filename, sep=',', header=0, index_col=0)
        artset_cls.from_dataframes(artset_descript=artset_descript, article_df=article_df, comment_df=comment_df)
    
    @classmethod
    def from_reddit_dfs(artset_cls, artset_descript, post_df, comment_df, scraper):
        """
        Creates an ArticleSet object from the two DataFrames as returned by the Reddit API and post-processed by the Scraper.query_reddit() method
        
        Parameters
        ----------
        artset_descript : str
            A description/label for to-be-created articleset
        post_df : pandas.DataFrame
            A DataFrame containing the posts (references to articles) as returned by the Reddit API
        comment_df : pandas.DataFrame
            A DataFrame containing the comments as returned by the Reddit API
        scraper : Scraper
            A Scraper object that can be used to parse the articles from the links referenced in Reddit posts

        Returns
        ----------
        artset : ArticleSet
            An ArticleSet object that represents the parsed articles and comments from the input DataFrames
        """
        artset = artset_cls(label=artset_descript, 
                            as_list=[], 
                            article_id_list=[])
        
        art_id_buffer, art_buffer = [], []
        for idx, df_row in post_df.iterrows(): #For each post
            
            #- - - - - - PARSE ARTICLE BODY 6 title: parse_article_from_link(url)
            parsed_article = scraper.parse_article_url(df_row["ext_link_url"])
            parsed_article["article_type"] = "posted_link"

            # - - - - - - - - 
            
            
            art = Article(article_id=parsed_article["url_id"],
                        article_type=parsed_article["article_type"], 
                        from_where=df_row["subreddit_name_prefixed"],
                        link=df_row["ext_link_url"],
                        post_id=[df_row["id"], ],
                        posted_by=[df_row["author_name"], ], #Note: This should be the author of the reddit submission/post, not the author of the article
                        date_posted=[df_row["created_utc"], ],
                        post_title=[df_row["title"], ], 
                        post_body=[df_row["selftext"], ],
                        date=parsed_article["date"],
                        author=parsed_article["author"], #Note: This should be the author of the article, not the author of the reddit submission/post
                        headline=parsed_article["headline"],
                        body=parsed_article["body"],
                        comments=[], 
                        comments_ids=[])
            
            buffer_id = -1 #Append article to buffer if it is not already there
            if art.article_id in art_id_buffer:
                buffer_id = art_id_buffer.index(art.article_id)
            else:
                art_id_buffer.append(art.article_id)
                art_buffer.append(art)

            p_cmt_id_buffer, cmt_buffer = [], []
            art_comments_df = comment_df.loc[comment_df['parent_post_id'] == df_row["id"]]
            for level in reversed(range(0, np.max(comment_df['depth'].astype(int)))): #For all levels
                comments = art_comments_df.loc[art_comments_df['depth'] == level]
                next_p_cmt_id_buffer, next_cmt_buffer = [], [] #Save comments from this level and overwrite the main (old) buffers just after completing this level
                for cmt_row_idx in comments.index:
                    cmt = Comment(comment_id=comments.loc[cmt_row_idx, "id"], 
                                    article_id=art_buffer[buffer_id].article_id, 
                                    date=comments.loc[cmt_row_idx, "created_utc"], 
                                    username=comments.loc[cmt_row_idx, "author_name"], 
                                    from_where=comments.loc[cmt_row_idx, "subreddit_name_prefixed"],
                                    body=comments.loc[cmt_row_idx, "body"],
                                    level=comments.loc[cmt_row_idx, "depth"],
                                    parent_comment_id=comments.loc[cmt_row_idx, "parent_comment_id"], 
                                    replies=[], 
                                    replies_ids=[])
                    
                    #Identify all replies to the current comment and store them in their parent
                    if cmt.comment_id in p_cmt_id_buffer: 
                        for buffer_idx in p_cmt_id_buffer.index(cmt.comment_id):
                            cmt.add_reply(cmt_buffer[buffer_idx])

                    next_p_cmt_id_buffer.append(cmt.parent_comment_id)
                    next_cmt_buffer.append(cmt)
                    
                #Reset buffers and refill for next level
                p_cmt_id_buffer, cmt_buffer = next_p_cmt_id_buffer, next_cmt_buffer
            
            for cmt in cmt_buffer: #Add all top level comments to the article
                art_buffer[buffer_id].add_comment(cmt)

        for art in art_buffer:
            artset.add_article(art)
        
        return artset
    
    @classmethod
    def from_youtube_query(artset_cls, artset_descript, query, scraper, max_results=100, sort_by="relevance"):
        artset = ArticleSet(label=artset_descript, 
                            as_list=[], 
                            article_id_list=[])
        
        c_vid_token = None
        vid_cntr = 0
        while c_vid_token != "last_page":
            
            
            c_vid_token, c_videos, total_num_results = yapi.video_search(q=query, 
                                                    order=sort_by, #Can be: relevance, date, rating, title, videoCount, viewCount
                                                    token=c_vid_token,
                                                    )

            
            for vid in c_videos:
                vid_cntr += 1
                print("Working on video ", vid_cntr, " of ", total_num_results,"...")
                art = Article(article_id=vid['id']['videoId'],
                            article_type="video", 
                            from_where="youtube",
                            link="https://www.youtube.com/watch?v="+vid['id']['videoId'],
                            post_id=None,
                            posted_by=None,
                            date_posted=None,
                            post_title=None, 
                            post_body=None,
                            date=vid['snippet']['publishedAt'],
                            author=vid['snippet']['channelTitle'], #Note: This should be the youtube channel name
                            headline=vid['snippet']['title'],
                            body=vid['snippet']['description'], #Interpret the video description as the article text
                            comments=[], 
                            comments_ids=[])
                
                p_cmt_id_buffer, cmt_buffer = [], []
                c_cmt_token = None #First page of comments corresponds to token=None
                
                while c_cmt_token != "last_page":
                    c_cmt_token, tld_comments, replies = yapi.video_comments_query(vid['id']['videoId'], token=c_cmt_token)
                    
                    for (cmt_level, comments) in [(1, replies), (0, tld_comments)]: #Reverse comment level iteration
                        next_p_cmt_id_buffer, next_cmt_buffer = [], [] #Save comments from this level and overwrite the main (old) buffers just after completing this level
                        for cmt_dict in comments:
                            cmt = Comment(comment_id=cmt_dict["id"], 
                                            article_id=cmt_dict["snippet"]["videoId"], 
                                            date=cmt_dict["snippet"]["publishedAt"],
                                            username=cmt_dict["snippet"]["authorDisplayName"],
                                            from_where="YouTube",
                                            body=cmt_dict["snippet"]["textDisplay"],
                                            level=cmt_level,
                                            parent_comment_id=cmt_dict["snippet"]["parentId"] if cmt_level == 1 else None, 
                                            replies=[], 
                                            replies_ids=[])
                            
                            #Identity all replies to the current comment and store them in their parent
                            if cmt.comment_id in p_cmt_id_buffer: 
                                for buffer_idx in list(locate(p_cmt_id_buffer, lambda x: x == cmt.comment_id)):   #p_cmt_id_buffer.index(cmt.comment_id):
                                    cmt.add_reply(cmt_buffer[buffer_idx])

                            next_p_cmt_id_buffer.append(cmt.parent_comment_id)
                            next_cmt_buffer.append(cmt)
                            
                        #Reset buffers and refill for next level
                        p_cmt_id_buffer, cmt_buffer = next_p_cmt_id_buffer, next_cmt_buffer


                
                for tl_cmt in cmt_buffer: #Add all top level comments to the article
                    art.add_comment(tl_cmt)

                artset.add_article(art)
            
            if vid_cntr > max_results:
                break

        return artset       
        

class Scraper:
    """
    Defines a scraper that queries, retrieves and parses article/comment data from the web
    """
    def __init__(self, reddit_api_wrapper="praw"):
        """
        Post-initialization; Sets the API wrapper to be used for querying the Reddit API
        
        Parameters
        ----------
        reddit_api_wrapper : str
            Reddit API wrapper name. Currently only supports "praw" or "pmaw"
            
        Returns
        ----------
        None
        """
        assert reddit_api_wrapper in ["praw", "pmaw"] #PRAW is the default wrapper, PMAW is a wrapper for the third party API Pushshift.io
        self.reddit_api_wrapper = reddit_api_wrapper
        
    def parse_article_url(self, url):
        """
        Given an URL try to extract and parse an article from it
        
        Parameters
        ----------
        url : str
            Some URL that (hopefully) points to an article 
            
        Returns
        ----------
        art_dict : dict
            A dictionary containing the parsed article data
        """
        #Generate a unique ID for the article
        url_id = hashlib.sha256(url.encode("utf-8")).hexdigest()
        
        #Define a return dictionary for the case of an invalid or (non-resolvable) redirect url
        art_dict = {"url_id": url_id,
                "article_type": "Unknown / Invalid or redirect URL",
                "link": url,
                "headline": "Unknown / Invalid or redirect URL",
                "author":  "Unknown / Invalid or redirect URL",
                "date": "Unknown / Invalid or redirect URL",
                "body": "Unknown / Invalid or redirect URL",
                }
        
        if validate_url(url) != None:
            if validate_url(url)[0]: #check whether url can be resolved or is damaged

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
                    art_dict = {"url_id": url_id,
                                "article_type": "web_article",
                                "link": url,
                                "headline": result["title"],
                                "author": result["author"],
                                "date": result["date"],
                                "body": result["text"],
                                }
        return art_dict
        
    
    
    async def query_from_subreddits(self,query, subreddits, post_limit_per_subreddit=1000, sort_by="relevance", min_comments_per_post=0,hostnames=None):
        """
        Searches in the Reddit API for posts that match the given query and are posted in the given subreddits. Gives flat dataframes containing the (meta)data of posts and their comments
        
        Parameters
        ----------
        query : str
            The query string to be passed to Reddit Search
        subreddits : list
            A list of subreddits which should be considered in search
        post_limit_per_subreddit : int (default: 1000)
            The maximum number of posts to be retrieved per subreddit; PRAW has a limit of 1000 posts per query
        sort_by : str (default: "relevance")
            The sorting method for the retrieved posts; can be "relevance", "new" or "comments"
        min_comments_per_post : int (default: 0)
            The minimum number of comments a post must have to be stored in the dataset
        hostnames : list[str] (default: None)
            A list of hostnames that should be considered in the search. If None, all hostnames are considered
        """
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
            orig_query = query
            
            # ----- Search and Retrieval ----- #

            combin_post_df, combin_comment_df = pd.DataFrame(), pd.DataFrame()
            
            post_buffer = []
            comment_buffer = []

            for subreddit in subreddits:
                print("Searching in subreddit: ", subreddit, "...")
                
                subreddit_inst = await reddit.subreddit(subreddit)

                cntr = 0
                
                hostname_chunks = ["",]
                if hostnames != None:
                    max_chunck_size = 10
                    hostname_chunks = [hostnames[i * max_chunck_size:(i + 1) * max_chunck_size] for i in range((len(hostnames) + max_chunck_size - 1) // max_chunck_size )]
                
                for hostname_chunk in hostname_chunks:
                    query = orig_query + " url:(" + " OR ".join(hostname_chunk) + ")"
                    print(query)
                    async for post_id in subreddit_inst.search(query=query, sort=sort_by,limit=post_limit_per_subreddit, time_filter="all"):
                        cntr += 1
                        
                        start_time_subm = time.time() #%
                        post = await reddit.submission(id=post_id)
                        if post.num_comments >= min_comments_per_post:
                            post_dict = post.__dict__
                            post_dict["author"] = post_dict["author"].name
                            post_buffer.append(post_dict)
                            
                            comments = post.comments
                            await comments.replace_more(limit=None)
                            for comment in comments.list():
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

            if len(post_buffer) == 0:
                return None, None
            else:
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

    def query_from_youtube(self, query, max_results=100, sort_by="relevance",dataset_descript="youtube data"):
        """
        Convenience wrapper function to query youtube data via an scraper instance.
        """
        return ArticleSet.from_youtube_query(artset_descript = dataset_descript, query = query, scraper=self, max_results = max_results, sort_by = sort_by).to_dataframes()



def main():
    pass

if __name__ == '__main__':
    main()
 