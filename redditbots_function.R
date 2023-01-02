# required package
# library(dplyr)
# library(tidyverse)
# library(lubridate)

subscriber_num <- function(source_dir, from_date, to_date) {
  df <- read_csv(source_dir)
  df <- df %>%
    dplyr::select(created_utc, subreddit_subscribers) %>%
    mutate(created_date = as_date(lubridate::as_datetime(created_utc))) %>%
    group_by(created_date) %>%
    summarise(max = max(subreddit_subscribers)) %>%
    filter(created_date >= as.Date(from_date) & created_date <= as.Date(to_date))
  # from = "2019-09-28", to = "2019-11-27"
  names(df)[2] <- "subscriber"
  return(df)
}

#multiple argument not permitted
findthresh <- function(df){
  thresh_ia <- df %>% group_by(post) %>%
    summarise(average_ia = mean(identity_attack)) %>%
    mutate(thresh_ia = mean(average_ia))
  thresh_insult <- df %>% group_by(post) %>%
    summarise(average_insult = mean(insult)) %>%
    mutate(thresh_insult = mean(average_insult))
  thresh_toxic <- df %>% group_by(post) %>%
    summarise(average_toxic = mean(toxicity)) %>%
    mutate(thresh_toxic = mean(average_toxic))
  return(thresh_ia)
  return(thresh_insult)
  return(thresh_toxic)
}
           
agg_data <- function(source_dir, s_toxic_thresh = 0.5, toxic_thresh, insult_thresh, ia_thresh, obscene_thresh = 0.5,
                     threat_thresh = 0.5) {
  df <- read_csv(source_dir)
  df <- df %>%
    mutate(created_date = as_date(lubridate::as_datetime(created_utc)),
           toxicity_f = ifelse(toxicity > toxic_thresh,1,0),
           s_toxicity_f = ifelse(severe_toxicity > s_toxic_thresh,1,0),
           identity_attack_f = ifelse(identity_attack > ia_thresh,1,0),
           insult_f = ifelse(insult > insult_thresh, 1, 0),
           obscene_f = ifelse(obscene > obscene_thresh, 1,0),
           threat_f = ifelse(threat > threat_thresh, 1,0)) %>%
    group_by(created_date) %>%
    summarise(
      toxicity = mean(toxicity),
      severe_toxicity = mean(severe_toxicity),
      obscene = mean(obscene),
      identity_attack = mean(identity_attack),
      insult = mean(insult), threat = mean(threat),
      obscene = mean(obscene),
      # sexual_explicit_m = mean(sexual_explicit),
      toxicity_p = mean(toxicity_f),
      severe_toxicity_p = mean(s_toxicity_f),
      obscene_p = mean(obscene_f),
      identity_attack_p = mean(identity_attack_f),
      insult_p = mean(insult_f),
      # exual_explicit_p = mean(sexual_explicit_flag),
      threat_p = mean(threat_f),
    )
  return(df)
}


agg_data2 <- function(source_dir, toxic_thresh=0.5, insult_thresh=0.5, ia_thresh=0.5) {
  df <- read_csv(source_dir)
  df <- df %>%
    mutate(created_date = as_date(as.POSIXct(created_utc,format="%m/%d/%Y %H:%M")),
           toxicity_f = ifelse(toxicity > toxic_thresh,1,0),
           identity_attack_f = ifelse(identity_attack > ia_thresh,1,0),
           insult_f = ifelse(insult > insult_thresh, 1, 0)) %>%
    group_by(created_date) %>%
    summarise(
      toxicity = mean(toxicity),
      severe_toxicity = mean(severe_toxicity),
      obscene = mean(obscene),
      identity_attack = mean(identity_attack),
      insult = mean(insult), threat = mean(threat),
      # sexual_explicit_m = mean(sexual_explicit),
      toxicity_p = mean(toxicity_f),
      #severe_toxicity_p = mean(severe_toxicity_flag),
      #obscene_p = mean(obscene_flag),
      identity_attack_p = mean(identity_attack_f),
      insult_p = mean(insult_f),
      # exual_explicit_p = mean(sexual_explicit_flag),
      #threat_p = mean(threat_flag)
    )
  return(df)
}

agg_data3 <- function(source_dir, toxic_thresh=0.5, insult_thresh=0.5, ia_thresh=0.5) {
  df <- read_csv(source_dir)
  df <- df %>%
    mutate(created_date = as_date(lubridate::as_datetime(created_utc)),
           toxicity_f = ifelse(toxicity > toxic_thresh,1,0),
           identity_attack_f = ifelse(identity_attack > ia_thresh,1,0),
           insult_f = ifelse(insult > insult_thresh, 1, 0)) %>%
    group_by(created_date) %>%
    summarise(
      toxicity = mean(toxicity),
      severe_toxicity = mean(severe_toxicity),
      obscene = mean(obscene),
      identity_attack = mean(identity_attack),
      insult = mean(insult), threat = mean(threat),
      # sexual_explicit_m = mean(sexual_explicit),
      toxicity_p = mean(toxicity_f),
      #severe_toxicity_p = mean(severe_toxicity_flag),
      #obscene_p = mean(obscene_flag),
      identity_attack_p = mean(identity_attack_f),
      insult_p = mean(insult_f),
      # exual_explicit_p = mean(sexual_explicit_flag),
      #threat_p = mean(threat_flag)
    )
  return(df)
}

add_relDate <- function(df, day_zero) {
  df$relative_day <- difftime(df$created_date, as.Date(day_zero, "%Y-%m-%d"), units = "days")
  df$post <- ifelse(df$relative_day <0, 0, 1)
  df$relative_day <- as.numeric(df$relative_day)
  return(df)
}


adjust_thresh <-  function (df, toxic_thresh = 0.5, s.toxic_thresh = 0.5,
                            obscene_thresh = 0.5, threat_thresh = 0.5, 
                            insult_thresh = 0.5, ia_thresh = 0.5) {
  df <- df %>%
    mutate_if(toxicity_f = ifelse(toxicity > toxic_thresh,1,0),
              s.toxity_f = ifelse(toxicity > toxic_thresh,1,0),
              identity_attack_f = ifelse(identity_attack > ia_thresh,1,0),
              insult_f = ifelse(insult > insult_thresh, 1, 0))
  return(df)
}

logodds <- function(x){log(x/(1-x))}

clean_comm.api <-  function(source_dir){
  df <-  read_csv(source_dir)
  df <- df %>%
    dplyr::select(total_awards_received, approved_at_utc, edited, banned_by,
                   removal_reason, link_id, replies, id, banned_at_utc,
                   gilded, author, created_utc, parent_id, score, mod_note, all_awardings,
                   subreddit_id, subreddit, body, awarders, name, downs, associated_award,
                   stickied, can_gild, top_awarded_type, permalink, num_reports, 
                   report_reasons, created, treatment_tags, controversiality, mod_reports,
                   ups, downs, author_fullname) %>%
    mutate(created_date = date(as_datetime(created_utc)))
  return(df)
}

within1month <- function(df, bot_epoch){
  df <-  df %>%
    mutate(created_date = as_datetime(created_utc )) %>%
    filter(created_utc > as_datetime(bot_epoch) + ddays(30) &
             created_utc <  as_datetime(bot_epoch) - ddays(30))
  return(df)
}

join_push.api <- function(df.push, df.api, report = TRUE){
  df.push <- df.push %>%
    dplyr::select(author, body, id, link_id, permalink) %>%
    rename(author.push = author, body.push = body, id.push = id,
           link_id.push = link_id, permalink.push = permalink)
  df.api <- df.api%>%
    dplyr::select(author, body, id, link_id, permalink)
  df.join <- left_join(df.push, df.api, by = c("id.push" = "id"), prefix = c(".x",".y"))
  if (report == TRUE){
    x1 = nrow(df.join %>% filter(body.push == "[deleted]", body == "[deleted]"))
    print(paste("Bot were deleted: ", x1))
    x2 = nrow(df.join %>% filter(body.push == "[removed]", body == "[removed]"))
    print(paste("Both were removed: ", x2))
    x3 = (nrow(df.join %>% filter(body.push  != "[deleted]", body == "[removed]")))
    print(paste0("Deleted on Reddit after Pushift scraped: ", x3))
    x4 = nrow(df.join %>% filter(body.push == "[deleted]", body != "[deleted]"))
    print(paste("Pushift deleted but recovered by now: ", x4))
    x5 = nrow(df.join %>% filter(body.push != "[removed]", body == "[removed]"))
    print(paste("Removed on Reddit after Pushift scraped: ", x5))
    x6 = nrow(df.join %>% filter(body.push == "[removed]", body != "[removed]"))
    print(paste("Pushift removed but recovered by now: ", x6))
    x7 = nrow(df.join %>% filter(author == "AutoModerator"))
    print(paste("AutoMod comments: ", x7))
  }
  df.join <- df.join %>%
    filter(body != "[removed]" & body != "[deleted]" & body.push != "[removed]" &
             body.push != "[deleted]" & author != "AutoModerator") %>%
    dplyr::select(id.push, author, body, link_id, permalink) %>%
    rename(id = id.push)
  return(df.join)
}

#not tested, translted from Python
clean_comm.push <- function(source_dir, bot_epoch, report = TRUE, save_dir){
  df <- read_csv(source_dir)
  df <-  df %>%
    unique() %>%
    dplyr::select(body, author, id, created_utc, retrieved_on, permalink,
                  parent_id, subreddit, subreddit_id) %>%
    filter(created_utc > as_datetime(bot_epoch) + ddays(30) &
             created_utc <  as_datetime(bot_epoch) - ddays(30)) %>%
    mutate(created_date = as_datetime(created_utc),
    post = ifelse(created_date > bot_epoch, 0, 1))
  if (report == TRUE) {
    tibb1 <- df %>% filter(body == "[deleted]" | body == "[removed]") %>%
      group_by(body,post) %>%
      summarise(N = n())
    print(as_tibble(tibb))
    tibb2 <- df %>% group_by(post)%>%
      summarise(N = n())
  }
  if (save_dir != FALSE){
    write_excel_csv(save_dir)
  }
  return(df)
}

find_metrics <- function(df.score, df.api, join_id){
  if (join_id=="permalink"){
    df <- left_join(df.score, df.api, by = c("permalink" = "permalink.score"))
  }
  if (join_id == "id"){
    rename(df.api, id.api = id)
    df <- left_join(df.score, df.api, by = c("id" = "id.api"))
  }
}