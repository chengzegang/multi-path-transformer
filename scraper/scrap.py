import json
from selenium import webdriver
from selenium.webdriver.common.by import By
from tqdm.auto import tqdm
import re
import yt_dlp
from sqlmodel import Field, SQLModel, create_engine, Session, select, func


class Video(SQLModel, table=True):
    id: int = Field(primary_key=True, default=None)
    url: str = Field(default=None, index=True)
    meta: bytes = Field(default=None)


engine = create_engine("sqlite:///video.db")
Video.metadata.create_all(engine)


def get_bv(url: str):
    pattern = r"BV(\w+)\/?"
    bvs = re.findall(pattern, url)
    if len(bvs) > 0:
        bv = bvs[0]
        bv = "BV" + bv
        return bv
    else:
        return None


def to_bilibili_url(bv: str):
    return "https://www.bilibili.com/video/" + bv


async def get_related_videos(driver: webdriver.Chrome):
    try:
        for card in driver.find_elements(By.CLASS_NAME, "card-box"):
            link = card.find_element(By.CLASS_NAME, "info").find_element(
                By.TAG_NAME, "a"
            )
            url = link.get_attribute("href")
            if url is None:
                continue
            bv = get_bv(url)
            if bv is None:
                continue
            video_url = to_bilibili_url(bv)
            title = link.get_attribute("title")
            yield bv, title, video_url
    except Exception as e:
        print(e)


async def rank_list_items(driver: webdriver.Chrome):
    for item in driver.find_elements(By.CLASS_NAME, "rank-item"):
        link = (
            item.find_element(By.CLASS_NAME, "content")
            .find_element(By.CLASS_NAME, "info")
            .find_element(By.TAG_NAME, "a")
        )
        url = link.get_attribute("href")
        if url is None:
            continue
        bv = get_bv(url)
        if bv is None:
            continue
        video_url = to_bilibili_url(bv)
        title = link.get_attribute("title")
        yield bv, title, video_url


async def get_board_items(driver: webdriver.Chrome):
    for item in driver.find_elements(By.CLASS_NAME, "board-item-wrap"):
        href = item.get_attribute("href")
        if href is None:
            continue
        bv = get_bv(href)
        if bv is None:
            continue
        video_url = to_bilibili_url(bv)
        title = item.get_attribute("alt")
        yield bv, title, video_url


async def get_video_cards(driver: webdriver.Chrome):
    video_cards = driver.find_elements(By.CLASS_NAME, "video-card")
    for card in video_cards:
        url = (
            card.find_element(By.CLASS_NAME, "video-card__content")
            .find_element(By.TAG_NAME, "a")
            .get_attribute("href")
        )
        bv = get_bv(url)
        if bv is None:
            continue
        video_url = to_bilibili_url(bv)
        title = (
            card.find_element(By.CLASS_NAME, "video-card__info")
            .find_element(By.TAG_NAME, "p")
            .text
        )
        yield bv, title, video_url


async def save_to_database(session: Session, url: str):
    info = await extract_info(url)
    if info is None:
        tqdm.write(f"failed to extract info: {url}")
        return None
    video = Video(url=url, meta=json.dumps(info))
    session.add(video)
    session.commit()
    session.flush()
    return url


async def login_videos(urls):
    with Session(engine) as session:
        driver = webdriver.Chrome()
        for base_url in tqdm(urls):
            driver.get(base_url)
            driver.implicitly_wait(5)
            coros = []
            async for bv, title, video_url in get_board_items(driver):
                tqdm.write(f"{title}: {video_url}")
                coros.append(asyncio.create_task(save_to_database(session, video_url)))

            async for bv, title, video_url in get_video_cards(driver):
                tqdm.write(f"{title}: {video_url}")
                coros.append(asyncio.create_task(save_to_database(session, video_url)))

            async for bv, title, video_url in rank_list_items(driver):
                tqdm.write(f"{title}: {video_url}")
                coros.append(asyncio.create_task(save_to_database(session, video_url)))
            await asyncio.gather(*coros)
        session.commit()
        session.flush()
    driver.close()


async def random_walk_scrap(num_walks: int = 5000):
    with Session(engine) as session:
        stmt = select(Video.url).order_by(func.random).limit(1)
        curr_url = session.exec(stmt).first()
        driver = webdriver.Chrome()
        for _ in tqdm(range(num_walks)):
            driver.get(curr_url)
            driver.implicitly_wait(5)
            urls = []
            coros = []
            async for bv, title, video_url in tqdm(get_related_videos(driver)):
                tqdm.write(f"{title}: {video_url}")
                if select(Video.url).filter_by(url=video_url).first() is not None:
                    continue
                coros.append(asyncio.create_task(save_to_database(session, video_url)))
                urls.append(video_url)
            urls = await asyncio.gather(*coros)
            urls = filter(lambda x: x is not None, urls)
            if len(urls) > 0:
                curr_url = urls[0]
            else:
                curr_url = session.exec(stmt).first()
        session.commit()
        session.flush()
        driver.close()


async def extract_info(url: str):
    options = {
        "quiet": True,
        "playlist_items": "1",
    }
    try:
        info = yt_dlp.YoutubeDL(options).extract_info(url, download=False)
        return info
    except Exception:
        return None


async def scrap(num_walks: int = 5000):
    # urls = [
    #    "https://www.bilibili.com/v/popular/history",
    #    "https://www.bilibili.com/v/popular/rank/all",
    #    "https://www.bilibili.com/",
    #    "https://www.bilibili.com/v/popular/drama/",
    # ] + [
    #    "https://www.bilibili.com/v/popular/weekly?num={}".format(i)
    #    for i in range(1, 240)
    # ]
    # await login_videos(urls)
    await random_walk_scrap(num_walks)


if __name__ == "__main__":
    import asyncio
 
    asyncio.run(scrap(5000))
