# Day7: 例外を統一して扱う
class ImageQualityError(Exception):
    """画像品質評価アプリ内で扱う例外（アプリ用のエラー）"""