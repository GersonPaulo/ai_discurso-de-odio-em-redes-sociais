
from controller import HateSpeechController

def main():
    print("Bem-vindo ao sistema de detecção de discurso de ódio!")
    data_path = input("Insira o caminho para o dataset (CSV): ")

    controller = HateSpeechController(data_path)
    controller.execute_pipeline()

    print("Pipeline concluído com sucesso!")

if __name__ == "__main__":
    main()
