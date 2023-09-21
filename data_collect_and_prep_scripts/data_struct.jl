using DataFrames
using PyCall
@pyimport praw # Reddit API Wrapper
@pyimport trafilatura #import fetch_url, extract, bare_extraction
@pyimport courlan #import validate_url, check_url, clean_url
@pyimport itertools

mutable struct Comment
    comment_id::String
    article_id::String
    date::Int # unix timestamp
    username::String
    from_where::String # "r/[subreddit]", "youtube", ...
    body::String
    level::Int
    parent_comment_id::String
    replies::Array{String, 1} 
    replies_ids::Array{String}
    Comment(;comment_id, article_id, date, username, from_where, body, level, parent_comment_id, replies= [], replies_ids= []) = new(comment_id, article_id, date, username, from_where, body, level, parent_comment_id, replies, replies_ids)
end


function add_reply(self::Comment, reply::Comment)
    push!(self.replies, reply)
    push!(self.replies_ids, reply.comment_id)
    return self
end

mutable struct Article
    article_id::String #Article ID on the host website or post ID on e.g. Reddit
    article_type::String # "web_article", "video_description", ...
    link::String 
    posted_by::String #Post author; only when applicable
    date_posted::Int #Post date, unix timestamp; when applicable
    post_title::String #Post title; when applicable
    post_body::String #Post body; when applicable
    date::String
    author::String
    headline::String
    body::String
    char_count::Int 
    comments::Array{Comment}
    comments_ids::Array{String}
    Article(;article_id, article_type, link, posted_by, date_posted, post_title, post_body, date, author, headline, body ,char_count = length(body), comments= [], comments_ids = []) = new(article_id, article_type, link, posted_by, date_posted, post_title, post_body, date, author, headline, body, comments, comments_ids)
end

    
function add_comment(self::Article, comment::Comment)
    push!(self.comments, comment)
    push!(self.comments_ids, comment.comment_id)
    return self
end

mutable struct ArticleSet
    label::String
    as_list::Array{Article}
    article_id_list::Array{Int}
    ArticleSet(;label, as_list= [], article_id_list= []) = new(label, as_list, article_id_list)
end
    
function add_article(self::ArticleSet, article::Article)
    push!(self.as_list, article)
    push!(self.article_id_list, article.article_id)
    return self
end

asdict(struct_inst) = Dict(key=>getfield(struct_inst, key) for key in fieldnames(typeof(struct_inst)))

function to_dataframes(self::ArticleSet)
    article_list = []
    comment_list = []
    for art in self.as_list
        art_dict = asdict(art)
        cmt_queue = art.comments
        while len(cmt_queue) != 0
            cmt_dict = asdict(popat!(cmt_queue, 1)) #Remove the saved comment from the queue
            for reply in cmt_dict["replies"]
                push!(cmt_queue, asdict(reply))
            end
            
            pop!(cmt_dict, "replies")
            push!(comment_list, cmt_dict)
        end
        pop!(art_dict, "comments")
        push!(article_list, art_dict)  
    end

    article_df = vcat(DataFrame.(article_list)...)
    comment_df = vcat(DataFrame.(comment_list)...)
    return article_df, comment_df
end


function from_dataframes(artset_descript, article_df, comment_df)
    artset = ArticleSet(label=artset_descript, 
                        as_list=[], 
                        article_id_list = [])
    
    for (idx, df_row) in enumerate(eachrow(article_df)) #For each post/article
        
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
        art_comments_df = filter(:parent_post_id => ==(art.article_id), comment_df) #Calls the function == partially
        for level in reverse(range(0, maximum(comment_df[!, "level"]))) #For all levels
            comments = filter(:level => ==(level), art_comments_df)
            next_p_cmt_id_buffer, next_cmt_buffer = [], [] #Save comments from this level and overwrite the main (old) buffers just after completing this level
            for cmt_row in eachrow(comments)
                cmt = Comment(comment_id=cmt_row["comment_id"], 
                                article_id=cmt_row["article_id"], 
                                date=cmt_row["date"], 
                                username=cmt_row["username"], 
                                from_where=cmt_row["from_where"],
                                body=cmt_row["body"],
                                level=cmt_row[ "level"],
                                parent_comment_id=cmt_row["parent_comment_id"], 
                                replies=[], 
                                replies_ids=cmt_row["replies_ids"])
                
                #Identity all replies to the current comment and store them in their parent
                if cmt.comment_id in p_cmt_id_buffer
                    for buffer_idx in findall(==(cmt.comment_id), p_cmt_id_buffer)
                        cmt = add_reply(cmt, cmt_buffer[buffer_idx])
                    end
                end
                push!(next_p_cmt_id_buffer, cmt.parent_comment_id)
                push!(next_cmt_buffer, cmt)
            
            end
            p_cmt_id_buffer, cmt_buffer = next_p_cmt_id_buffer, next_cmt_buffer
        end

        for cmt in cmt_buffer #Add all top level comments to the article
            art = add_comment(art, cmt)
        end

        artset = add_article(artset, art)
    end

    return artset
end
"""
    def save_to_csv(self, filepath):
      
        article_df, comment_df = self.to_dataframes()
        article_filename, comment_filename = filepath + "_articles.csv", filepath + "_comments.csv"
        article_df.to_csv(article_filename, sep=",", header=True, index=True, index_label=None)
        comment_df.to_csv(comment_filename, sep=",", header=True, index=True, index_label=None)
    	
    def load_from_csv(artset_cls, filepath, artset_descript):
    
        article_filename, comment_filename = filepath + "_articles.csv", filepath + "_comments.csv"
        article_df = pd.read_csv(article_filename, sep=",", header=0, index_col=0)
        comment_df = pd.read_csv(comment_filename, sep=",", header=0, index_col=0)
        artset_cls.from_dataframes(artset_descript=artset_descript, article_df=article_df, comment_df=comment_df)
"""

mutable struct Scraper
    reddit_api_wrapper::String # The reddit API wrapper to use. Currently only "praw" is supported.
    Scraper(;reddit_api_wrapper="praw") = new(reddit_api_wrapper)
end
        
function parse_article_url(self::Scraper; url::String)
    #Define a return dictionary for the case of an invalid or (non-resolvable) redirect url
    art_dict = Dict("article_type" => "web_article",
            "link" => url,
            "headline" => "Unknown / Invalid or redirect URL",
            "author" =>  "Unknown / Invalid or redirect URL",
            "date" => "Unknown / Invalid or redirect URL",
            "body" => "Unknown / Invalid or redirect URL",
            )
    
    if courlan.validate_url(url)[1] && (courlan.check_url(url)[end] == "youtube.com" || check_url(url)[end] == "youtu.be") #Check whether url is a youtube video
    #TODO: Implement youtube video parsing via youtube API
    elseif courlan.validate_url(url)[1] #check whether url can be resolved or is damaged
        url = courlan.clean_url(url)
        downloaded = trafilatura.fetch_url(url)

        result = trafilatura.bare_extraction(downloaded, 
                                    :url=>url, 
                                    #include_comments=True,
                                    #include_formatting=False,
                                    :favor_precision=>true, #favor recall
                                    :with_metadata=>true,
                                    #deduplicate=True, 
                                    :only_with_metadata=>true,
                                    #url_blacklist=None,
                                    :as_dict=>true,
                                    )

        if result #If the article was successfully parsed; e.g. no redirect url
            #Replace the return dictionary with the parsed data
            art_dict = Dict("article_type"=> "web_article",
                        "link" => url,
                        "headline" => result["title"],
                        "author" => result["author"],
                        "date" => result["date"],
                        "body" => result["text"],
                        )
        end
    end
    return art_dict
end

function query_from_subreddits(self::Scraper; query::String, subreddits::Array{String,1}, post_limit_per_subreddit::Int=1000, sort_by::String="relevance")
    reddit = praw.Reddit(
            client_id="DIAteHPvd-6mPi8WMCykog",
            client_secret="PckY-T0xsvax4OgrWWvBkTxVIwvapg",
            #:password=>"bonGux-gyrro0-zuvhuv",
            user_agent="python:DIAteHPvd-6mPi8WMCykog:0.1 scraper (by u/SciPhaM) ",
            #:username=>"SciPhaM",
            )
    
    if self.reddit_api_wrapper == "praw" #PRAW Based Search and Retrieval
        @assert post_limit_per_subreddit <= 1000 #PRAW has a limit of 1000 posts per query
        @assert sort_by in ["relevance", "new", "comments"]
        
        # ----- Search and Retrieval ----- #

        combin_post_df, combin_comment_df = DataFrame(), DataFrame()
        
        query_result_list = []
        for subreddit in subreddits
            print("Searching in subreddit: ", subreddit, "...")
            query_result = reddit.subreddit(subreddit).search(query=query,
                                                        sort=sort_by, # "comments",
                                                        limit=post_limit_per_subreddit,
                                                        time_filter="all",
                                                        )
            push!(query_result_list, query_result)
            sleep(6)
        end
        query_result_concat = collect(itertools.chain(query_result_list...))

        post_buffer = []
        comment_buffer = []

        cntr = 0
        
        for post_id in query_result_concat
            cntr += 1
            
            post = reddit.submission(id=post_id)
            if post.num_comments > 0
                post_dict = post.__dict__ 
                post_dict["author"] = post_dict["author"].name
                push!(post_buffer, post_dict)

                post.comments.replace_more(limit=nothing)
                for comment in post.comments.list()
                    comment_dict = comment.__dict__
                    
                    #if comment_dict["body"] != "[removed]":  #Some comments were deleted, don"t include them
                    comment_dict["_submission"] = comment_dict["_submission"].id
                    try
                        comment_dict["author"] = comment_dict["author"].name
                    catch #If author or comment was deleted, don"t include it in the data
                        continue
                    end

                    comment_dict["_replies"] = tuple([reply.id for reply in comment_dict["_replies"]]...)
                    push!(comment_buffer, comment_dict)

                end
                print("Donxe with processing post $cntr")
            end
        end
        print(size(comment_buffer))

        #Store as a DataFrames for convenience
        post_buffer = vcat(DataFrame.(post_buffer)...)
        comment_buffer = vcat(DataFrame.(comment_buffer)...)

        # ----- Post Processing ----- #
        #Only pass on the relveant features
        
        post_features = [ "id", "permalink", "subreddit_id", "subreddit_name_prefixed","author", "author_fullname", "title","created_utc", "is_self","selftext","selftext_html" , "url", "media", "num_comments"]
        comment_features = ["id", "permalink","subreddit_id","subreddit_name_prefixed", "author", "author_fullname","_submission", "link_id", "parent_id" ,"created_utc", "body", "body_html", "depth" , "_replies"]
        
        post_df = post_buffer[!, post_features]
        comment_df = comment_buffer[!, comment_features]

        rename!(post_df, [:author => :author_name, :author_fullname => :author_id, :url => :ext_link_url])
        rename!(comment_dfm, [:author => :author_name, :author_fullname => :author_id, :_submission => :parent_post_id, :parent_id => :parent_comment_id, :_replies => :replies_ids])

        post_df[!,"subreddit_id"] = [split(item, "_")[end] for item in copy(post_df[!, "subreddit_id"])] #Remove the t5_ prefix
        post_df[!,"author_id"] = [split(item, "_")[end] for item in copy(post_df[!, "author_id"])] #Remove the t2_ prefix

        comment_df[!,"subreddit_id"] = [split(item, "_")[end] for item in copy(comment_df[!,"subreddit_id"])] #Remove the t5_ prefix
        comment_df[!,"author_id"] = [split(item, "_")[end] for item in copy(comment_df[!, "author_id"])] #Remove the t2_ prefix

        comment_df[!,"parent_comment_id"] = [split(item, "_")[1] == "t3" ? split(item, "_")[end] : nothing for item in copy(comment_df[!, "parent_comment_id"]) ] #Remove the t3_ prefix and substitue t1_ ids with None (for top level comments)

        combin_post_df = vcat([combin_post_df, post_df]...)
        combin_comment_df = vcat([combin_comment_df, comment_df]...)
        
        return (combin_post_df, combin_comment_df)
    end
end

function from_reddit_dfs(artset_descript::String, post_df::DataFrame, comment_df::DataFrame, scraper::Scraper)
 

    artset = ArticleSet(label=artset_descript, 
                        as_list=[], 
                        article_id_list=[])
            
    for (idx, df_row) in enumerate(eachrow(post_df)) #For each post/article
        
        #- - - - - - PARSE ARTICLE BODY 6 title: parse_article_from_link(url)
        parsed_article = parse_article_url(scraper, df_row["ext_link_url"])
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
        art_comments_df = comment_df.loc[comment_df["parent_post_id"] == art.article_id]
        for level in reverse(range(0, maximum(convert.(Int, comment_df["depth"])))) #For all levels
            comments = filter(:depth => ==(level), art_comments_df)
            next_p_cmt_id_buffer, next_cmt_buffer = [], [] #Save comments from this level and overwrite the main (old) buffers just after completing this level
            for cmt_row in eachrow(comments)
                cmt = Comment(comment_id=comments["id"], 
                                article_id=cmt_row["parent_post_id"], 
                                date=cmt_row["created_utc"], 
                                username=cmt_row["author_name"], 
                                from_where=cmt_row["subreddit_name_prefixed"],
                                body=cmt_row["body"],
                                level=cmt_row["depth"],
                                parent_comment_id=cmt_row["parent_comment_id"], 
                                replies=[], 
                                replies_ids=[])
                
                #Identity all replies to the current comment and store them in their parent
                if cmt.comment_id in p_cmt_id_buffer
                    for buffer_idx in findall(==(cmt.comment_id), p_cmt_id_buffer)
                        cmt = add_reply(cmt, cmt_buffer[buffer_idx])
                    end
                end
                push!(next_p_cmt_id_buffer, cmt.parent_comment_id)
                push!(next_cmt_buffer, cmt)
            end
            #Reset buffers and refill for next level
            p_cmt_id_buffer, cmt_buffer = next_p_cmt_id_buffer, next_cmt_buffer
        end

        for cmt in cmt_buffer #Add all top level comments to the article
            art = add_comment(art, cmt)
        end

        artset = add_article(artset, art)

    end

    return artset
end
