from xml.dom.pulldom import START_ELEMENT
import pandas as pd
import numpy as np
import scipy.stats as sts
from detoxify import Detoxify
import datetime as dt
import time
import requests
from pmaw import PushshiftAPI
import praw


#### Comments: Clean and Detox

def get_epoch(bot_epoch, duration):
    before = int((dt.datetime.fromtimestamp(bot_epoch)+dt.timedelta(days = duration)).timestamp())
    after = int((dt.datetime.fromtimestamp(bot_epoch)-dt.timedelta(days = duration)).timestamp())
    return before, after

def num_case(df):
    num_post = len(df[df['post']==1])
    num_pre = len(df[df['post']==0])
    print(f'Pre: {num_pre}, Post: {num_post}, Total: {num_pre+num_post}')

def comm_report(df):
    print(f'This df originally has {len(df)} comments.') #print no of comments
    num_pre = len(df[df['post']==0])
    num_post = len(df[df['post']==1])
    print(f'Pre:{num_pre}, Post:{num_post}') #print num of pre and post
    num_del = len(df[(df['body'] == '[deleted]') | (df['body'] == '[removed]')])
    print(f'{num_del} comments were deleted/removed.')  #print no of del comments
    num_automod = len(df[df['author'] =='AutoModerator']) 
    print(f'Automod posted {num_automod} comments.') #print comments by AutoMod

def clean_comments_df(df, save_dir, bot_epoch = False, duration=False, verbose = True):
    df_nodup = df.drop_duplicates()
    #read files and choose relevant vars
    df_id = df_nodup.reindex(columns = ['body','author','created_utc','retrieved_on','permalink','parent_id','subreddit','subreddit_subscribers','score','post'])
    df_short = df_id[['body','author','created_utc','retrieved_on','permalink','parent_id','subreddit','subreddit_subscribers','score','post']]
    #filter comments within 60 days of bot implementation
    if bot_epoch != False & duration != False:
        before, after = get_epoch(bot_epoch, duration)
        df_within = df_short[(df_short['created_utc'] >=  after) & (df_short['created_utc'] <= before)]
    #change epoch time to human time
    df_within['created_utc'] = pd.to_datetime(df_within['created_utc'], unit='s')
    df_within['retrieved_on'] = pd.to_datetime(df_within['retrieved_on'], unit='s')
    df_within['retrieved_on'] = pd.to_datetime(df_within['retrieved_on'], unit='s')
    df_within['updated_utc'] = pd.to_datetime(df_within['retrieved_on'], unit='s')
    #filter out deleted and removed comments
    df_nodel = df_within[(df['body'] != '[deleted]') & (df_within['body'] != '[removed]') & (df_within['author'] !='AutoModerator')]
    df_out = df_nodel.dropna(subset='body')
    #moare report
    if verbose == True:
        comm_report(df)
        num_case(df)
    #write csv
    df_out.to_csv(save_dir,encoding = 'utf-8-sig')
    return df

def detox_loop(df,model,file_out = False):
    detox = Detoxify(model, device='cuda')
    i = 0
    n = len(df)
    df_res = pd.DataFrame()
    tic = time.time()
    while i < n:
        j = min (i+100,n)
        res = detox.predict(df[i:j])
        f = pd.DataFrame(res,df[i:j]).round(5)
        df_res = df_res.append(f)
        if i%10000 == 0:
            time.sleep(.1)
            print('Time elapsed: ', round(time.time() - tic),' secs, i = ', i)
        i +=100
        if i%500000 == 0 & file_out != False:
            res_out = pd.DataFrame(df_res)
            res_out.to_csv(file_out + '_res_' + str(i) + '.csv', encoding = 'utf_8_sig')
            res_out = []
    return df_res

def flaten(df):
    df=list(df['body'].values.flatten())
    return df

def detox_df(df,model, save_dir, file_out):
    flaten_df = flaten(df=df)
    detoxdf = detox_loop(df=flaten_df,model=model, file_out=file_out)
    res = pd.concat([df.reset_index(drop=True),detoxdf.reset_index(drop=True)], axis = 1)
    if save_dir != False:
        res.to_csv(save_dir,encoding = 'utf-8-sig')
    return res

# praw enhancement
reddit = praw.Reddit(
 client_id='9wXh5oa4cfW07_eUn5Hu3A',
 client_secret='m3GnhaEvbM5LGCWX3BMghLCatOLN3g',
 user_agent=f'python: PMAW request enrichment (by u/softyarn)'
)

api_praw = PushshiftAPI(praw=reddit)

# pmaw set up
api = PushshiftAPI()

# pmaw comments

def get_epoch(bot_epoch, duration):
    after = int((dt.datetime.fromtimestamp(bot_epoch)+dt.timedelta(days = duration)).timestamp())
    before = int((dt.datetime.fromtimestamp(bot_epoch)-dt.timedelta(days = duration)).timestamp())
    return before, after


def fetch_comments(subr,save_dir, bot_epoch, duration, limit):
    before, after = get_epoch(bot_epoch, duration)
    tic = time.time()
    res = []
    comments = api.search_comments(subreddit=subr, limit=limit, before=before, after=after, safe_mem = True)
    res = [c for c in comments]
    df = pd.DataFrame(res)
    save(save_dir,response=df)
    return df

def fetch_subm(subr,save_dir, bot_epoch, duration, limit):
    before, after = get_epoch(bot_epoch, duration)
    tic = time.time()
    res = []
    comments = api.search_submissions(subreddit=subr, limit=limit, before=before, after=after, safe_mem = True)
    res = [c for c in comments]
    df = pd.DataFrame(res)
    save(save_dir,response=df)
    return df

def save(save_dir,response):
    response.to_csv(save_dir, header=True, index=False, columns=list(response.axes[1]),  encoding = 'utf-8-sig')


# Reddit API (no wrapper)
def clean_id(source_dir, comment_bool):
    df = pd.read_csv(source_dir)
    df = df[["id","body"]]
    if comment_bool == True:
        df['id'] ='t1_' + df['id'].astype(str)
    else:
        df['id'] ='t3_' + df['id'].astype(str)
    df["note"] = np.where(df["body"] == "[removed]", "removed", np.where(df["body"] == "[deleted]", "deleted", "text"))
    return df

def get_comments(df, headers) :
    tic = time.time()
    global res_df
    res_df = []
    i = 0
    while i < len(df["id"]):
        j = min(len(df["id"]), i + 40)
        x = ",".join(df["id"][i:j])
        comments = requests.get('https://oauth.reddit.com/api/info',headers=headers,params={'id': x}).json()
        for k in range(j-i):
            comment_df = pd.DataFrame.from_dict(comments.get("data").get("children")[k].get("data"), orient = "index").transpose()
            res_df.append(comment_df)
        i += 40
        if i%300 == 0:
            time.sleep(2)
            #now = time.localtime(time.time())
            print('Time elapsed: ', round(time.time() - tic),' secs, i = ',i)
            #print(time.strftime("Time now: %HH:%M:%S",now),f', i = {i}')
    return pd.concat(res_df)

def get_submissions(df, headers):
    tic = time.time()
    global res_df
    res_df = pd.DataFrame()
    i = 0
    while i < len(df["id"]):
        j = min(len(df["id"]), i + 40)
        x = ",".join(df["id"][i:j])
        submissions = requests.get('https://oauth.reddit.com/by_id/names',headers=headers,params={'names': x}).json()
        for k in range(20):
            submission_df = pd.DataFrame.from_dict(submissions.get("data").get("children")[k].get("data"), orient = "index").transpose()
            res_df.append(submission_df)
        i += 40
        if i%300 == 0:
            time.sleep(2)
            print('Time elapsed: ', round(time.time() - tic),' secs, i = ',i)
    return pd.concat(res_df)  

# Get Authors
def fetch_comments (subreddit, bot_epoch, before, after, duration, limit = 1000000000):
    before = int((dt.datetime(bot_epoch)-dt.timedelta(days = duration)).timestamp())
    after = int((dt.datetime(bot_epoch)+dt.timedelta(days = duration)).timestamp())
    comments = api.search_comments(subreddit = subreddit, limit = limit, before = before, after = after)
    df = pd.DataFrame(comments)
    df['post']=1
    df.loc[df['created_utc'] < bot_epoch, 'post'] = 0
    return df

def flatten_author(df):
    authors = df.loc[:,'author']
    authors_unique = np.unique(authors)
    authors_str = ','.join(authors_unique)
    return authors_str

def author_list(df):
    authors = df.loc[:,'author']
    authors_unique = np.unique(authors)
    return authors_unique

def get_epoch(bot_epoch, duration):
    before = int((dt.datetime.fromtimestamp(bot_epoch)+dt.timedelta(days = duration)).timestamp())
    after = int((dt.datetime.fromtimestamp(bot_epoch)-dt.timedelta(days = duration)).timestamp())
    return before, after

def author_comments(df, bot_epoch, duration, limit = 10000000000, enhance = True):
    before, after = get_epoch(bot_epoch, duration) 
    #creating loop so that only 100 authors per praw
    tic = time.time()
    global res_df
    res_df = []
    authors = author_list(df)
    i = 0
    while i < len(authors):
        j = min(i+100, len(authors))
        x = ','.join(authors[i:j])
        comments = api.search_comments(author = x, limit = limit, before = before, after = after)
        comments_list = [c for c in comments]
        res_df.append(comments_list)
        i +=100
    if i%500 == 0:
        time.sleep(1)
        print('Time elapsed: ', round(time.time() - tic),' secs, i = ',i)
    return pd.concat(res_df)

def get_epoch(bot_epoch, duration):
    before = int((dt.datetime.fromtimestamp(bot_epoch)+dt.timedelta(days = duration)).timestamp())
    after = int((dt.datetime.fromtimestamp(bot_epoch)-dt.timedelta(days = duration)).timestamp())
    return before, after

def within60days(df, bot_epoch, duration):
    before, after = get_epoch(bot_epoch = bot_epoch, duration = duration)
    res = df.loc[(df['created_utc'] >= after) & (df['created_utc'] <= before)]
    return res

# Get list of authors 
def author_list(df):
    nomod = df.loc[(df['author'] != "AutoModerator") & (df['author'] != '[deleted]')]
    authors = nomod.loc[:,'author']
    authors_unique = np.unique(authors)
    return authors_unique

# Get comments based on author id
def author_comments(df, bot_epoch, duration, limit = 10000000000):
    before, after = get_epochdate(bot_epoch, duration) 
    #creating loop so that only 100 authors per praw
    tic = time.time()
    global res
    res = []
    authors = author_list(df)
    i = 0
    while i < len(authors):
        j = min(i+2, len(authors))
        x = ','.join(authors[i:j])
        comments = api.search_comments(author = x, limit = limit, before = before, after = after, filter = ['id', 'banned_at_utc', 'mod_reason_title', 'author', 'created_utc', 'parent_id', 'subreddit_id', 'body'], mem_safe= True)
        res += [c for c in comments]
        if i%500 == 0:
            res_out = pd.DataFrame(res)
            res_out.to_csv('author_comments_' + str(i/1000) + '.csv', encoding = 'utf_8_sig')
            res = []
        i +=2
        if i%100 == 0:
            time.sleep(1)
            print('Time elapsed: ', round(time.time() - tic),' secs, i = ',i)
    res_out = pd.DataFrame(res)
    res_out.to_csv('author_comments_Finale.csv', encoding = 'utf_8_sig')   
    return res_out

