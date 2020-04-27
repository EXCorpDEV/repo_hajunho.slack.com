//
//  jhSceneTimeLine.swift
//  com.hajunho.swift-graph
//
//  Created by Junho HA on 2018. 10. 24..
//  Copyright © 2018년 hajunho.com. All rights reserved.
//

import UIKit

class jhSceneTimeLine : jhScene {
    
    private var mPanels : [jhPanel<jhSceneTimeLine>] = Array<jhPanel>()
    
    var currentTime: Date = Date(timeIntervalSince1970: 1531180740) //TODO: should be changed to init() creates with current time.
    
    var endTime: Date {
        get {
            return Date(timeIntervalSince1970: currentTime.timeIntervalSince1970 - Double(GS.s.sceneWidthByTime))
        }
    }
    
    override func createPanels(s : jhScene, withHeightRatios: ratioNtype...) {
        
        var panel : jhPanel<jhSceneTimeLine>? = nil
        var vHeight : CGFloat = 0.0
        
        "".pwd(self)
        GS.s.jhSceneHeight = 0.0
        
        for rnt in withHeightRatios {
            print("**************CTIME****************")
            if(GS.s.logLevel.contains(.graphPanel)) { print("createPanels(withHeightRatios: CGFloat...)", rnt)}
            
            assert(!(rnt.ratio < 0.1 || rnt.ratio > 10.0), "heightRation Range is 0.1~10.0")
            
            vHeight = rnt.ratio * 0.1 * self.jhSceneFrameHeight
            GS.s.jhSceneHeight! += GS.s.jhPSpacing
            
            panel = jhGraphBuilder<jhSceneTimeLine>()
                .type(rnt.type)
                .frame(0, GS.s.jhSceneHeight!, jhSceneFrameWidth*4, vHeight)
                .scene(self)
                .build()
            GS.s.jhSceneHeight! += vHeight
            
            if GS.s.logLevel.contains(.graphPanel) {
                print("jhScene_addPanel_mHeightStack =", vHeight, "\n y = \(GS.s.jhSceneHeight!) heightRatio = \(rnt)")
            }
            
            if GS.s.logLevel.contains(.network2) {
                print("ctime in jhSceneTimeLine_createPanels", (panel?.superScene as? jhSceneTimeLine)?.currentTime)
            }
            
            panel!.backgroundColor = UIColor.white
            mPanels.append(panel!)
            panel = nil
        }
    }
    
    var xRatioByTime : CGFloat {
        get {
            return jhDraw.ARQ / GS.s.sceneWidthByTime
        }
    }
    
    override func jhColor(r:CGFloat , g:CGFloat , b:CGFloat , a:Float) -> CGColor {
        return  UIColor(red: r / 255.0, green: g / 255.0, blue: b / 255.0, alpha: r).cgColor
    }
    
    override func drawScene() {
        drawPanels()
    }
    
    override func drawPanels() {
        
        self.imageContentMode = .widthFill
        self.initialOffset = .center
        
        let ret : UIView = UIView(frame: CGRect(x: 0, y: 0, width: self.contentSize.width, height: self.contentSize.height))
        
        for x in mPanels {
            //            super.addSubview(x)
            ret.addSubview(x)
        }
        self.display(view: ret)
    }
}
