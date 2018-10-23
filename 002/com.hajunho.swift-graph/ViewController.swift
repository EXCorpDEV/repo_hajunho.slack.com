//
//  ViewController.swift
//  com.hajunho.swift-graph
//
//  Created by Junho HA on 2018. 10. 16..
//  Copyright © 2018년 hajunho.com. All rights reserved.
//

import UIKit
import SnapKit

class ViewController: UIViewController {

    @IBOutlet var safeArea: UIView!
    
    override func viewDidLoad() {
        super.viewDidLoad()

        
//        jhServerTest.init()
        
        
        // Do any additional setup after loading the view.
        
        let scrollView : UIScrollView = UIScrollView(frame: CGRect(x: 0, y: 0, width: UIScreen.main.bounds.width, height: UIScreen.main.bounds.height))
        
        scrollView.contentSize = CGSize(width: safeArea.frame.width, height: 894) //TODO: contents size
        scrollView.setContentOffset(CGPoint(x: 0, y: 200), animated: true)
        scrollView.isUserInteractionEnabled = true
        scrollView.translatesAutoresizingMaskIntoConstraints = true
        scrollView.maximumZoomScale = 4.0
        scrollView.minimumZoomScale = 0.1
        
        let graphPanel : jhScene = addGraph()
//
        let motherOfGraphScene : UIView = UIView(frame: CGRect(x: 0, y: 0, width: scrollView.bounds.width, height: scrollView.bounds.height))
        motherOfGraphScene.addSubview(graphPanel)
        scrollView.addSubview(motherOfGraphScene)
//
//        motherOfGraphScene.snp.makeConstraints ({ (make) in
//            make.left.equalTo(scrollView.snp.left)
//            make.width.equalTo(2000) //TODO:
//            make.height.equalTo(3000)
//            make.top.equalTo(scrollView.snp.bottom).offset(1)
//        })
        
        scrollView.isScrollEnabled = true
//        motherOfGraphScene.resignFirstResponder()
        safeArea.addSubview(scrollView)
    }
    

    /*
    // MARK: - Navigation

    // In a storyboard-based application, you will often want to do a little preparation before navigation
    override func prepare(for segue: UIStoryboardSegue, sender: Any?) {
        // Get the new view controller using segue.destination.
        // Pass the selected object to the new view controller.
    }
    */
    
    fileprivate func addGraph() -> jhScene {
        
        var scene: jhScene? = nil
        
        scene = jhScene(frame: CGRect(x: 0, y: 0, width: UIScreen.main.bounds.width, height: 488)) //TODO: sum of Panel size
        
        scene!.contentSize = CGSize(width: scene!.frame.width*4, height: scene!.frame.height+100) //TODO:
        scene!.isUserInteractionEnabled = true
        scene!.translatesAutoresizingMaskIntoConstraints = true
        scene!.maximumZoomScale = 4.0
        scene!.minimumZoomScale = 0.1
        scene!.isScrollEnabled = true
        
        scene!.backgroundColor = UIColor.white
        
        scene!.createPanels(withHeightRatios: ratioNtype(ratio: 5, type: graphType.LINE), ratioNtype(ratio: 3, type: graphType.BAR), ratioNtype(ratio: 2, type: graphType.BAR))
        scene!.drawScene()
        
        let ret : UIView = UIView(frame: CGRect(x: 0, y: 0, width: scene!.bounds.width, height: scene!.bounds.height))
        ret.addSubview(scene!)
        //        return ret
        return scene!
    }
    

}
