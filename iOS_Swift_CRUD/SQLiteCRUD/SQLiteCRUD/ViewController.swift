//
//  ViewController.swift
//  SQLiteCRUD
//
//  Created by Junho HA on 2021/05/28.
//

import UIKit

class ViewController: UIViewController {
    
    override func viewDidLoad() {
        super.viewDidLoad()
        // Do any additional setup after loading the view.
    }
    
    @IBAction func onClickButton(_ sender: Any) {
        debugPrint("onClickButton")
        let acroView = self.storyboard?.instantiateViewController(withIdentifier: "acroxib") as! ViewControllerACRO
        acroView.modalPresentationStyle = .fullScreen
//        acroView.modalPresentationStyle = .currentContext
//        acroView.modalPresentationStyle = .none //ERROR
//        acroView.modalPresentationStyle = .overCurrentContext
//        acroView.modalPresentationStyle = .overFullScreen
        
//        acroView.modalPresentationStyle = .automatic
//        acroView.modalPresentationStyle = .popover
//        acroView.modalPresentationStyle = .pageSheet
        
        self.present(acroView, animated: true, completion: nil)
    }
}

